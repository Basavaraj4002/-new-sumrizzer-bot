 # --- Python/FastAPI Backend Code (main.py) ---
# Uses PyMuPDF, includes key alignment for frontend compatibility

import io
import re
import logging
import traceback
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import json

# Configure logging - Set to DEBUG for detailed extraction info
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Marks Summarizer API (PyMuPDF)", version="1.4.1") # Incremented version

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Global exception handler middleware
@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except HTTPException as he:
        logger.warning(f"HTTP Exception caught: Status={he.status_code}, Detail={he.detail}")
        raise he
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception for request {request.method} {request.url.path}: {str(e)}\n{error_traceback}")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred processing the request.", "error": str(e)},
        )

# --------------------------------------------------------------------------
# Data Extraction Function using PyMuPDF (fitz) - Remains the same
# --------------------------------------------------------------------------
def extract_marks_fitz(pdf_content: bytes):
    """
    Extracts student marks from PDF content using PyMuPDF (fitz).
    Relies on regex matching on the extracted text per page.
    Includes USN cleanup and duplicate handling.
    """
    logger.info("Starting PDF extraction process using PyMuPDF (fitz).")
    all_students = []
    processed_usns = set()
    doc = None
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        logger.info(f"PDF opened successfully with fitz. Pages: {doc.page_count}")
        pattern = r'(\d+)\s+([A-Z0-9]{10})\s+(?:[A-Z\s]+)?\s*(\d+)\s*\((?:TH|Theory)\)\s*[,;]?\s*(\d+)\s*\((?:PR|Practical|PRA|PRAC)\)'

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            page_id = f"Page {page_num + 1}"
            if not text: logger.warning(f"{page_id}: No text extracted."); continue
            logger.debug(f"{page_id}: Text length: {len(text)}")

            matches = re.finditer(pattern, text, re.IGNORECASE)
            found_on_page = 0
            for match_num, match in enumerate(matches, 1):
                try:
                    usn = match.group(2).strip().upper()
                    if usn in processed_usns: continue # Skip duplicates

                    sl_no_val = int(match.group(1))
                    theory_val = int(match.group(3))
                    practical_val = int(match.group(4))

                    if not (0 <= theory_val <= 100 and 0 <= practical_val <= 100):
                        logger.warning(f"{page_id} Match {match_num}: USN={usn} - Invalid marks (TH={theory_val}, PR={practical_val}). Skipping.")
                        continue

                    student = {'SL_NO': sl_no_val, 'USN': usn, 'Theory': theory_val, 'Practical': practical_val}
                    all_students.append(student)
                    processed_usns.add(usn)
                    found_on_page += 1
                    logger.debug(f"{page_id} Match {match_num}: Added USN={usn}, TH={theory_val}, PR={practical_val}")

                except Exception as match_err:
                    logger.warning(f"{page_id} Match {match_num}: Error processing match: {match_err}. Groups: {match.groups()}")

            if found_on_page > 0: logger.info(f"{page_id}: Found {found_on_page} valid records.")
            else: logger.info(f"{page_id}: No records matched pattern.")

        if not all_students: raise ValueError("No valid student records extracted.")
        logger.info(f"Successfully extracted {len(all_students)} unique records using PyMuPDF.")
        return pd.DataFrame(all_students)

    except ValueError as ve: raise HTTPException(status_code=400, detail=str(ve))
    except fitz.fitz.FileDataError as fitz_err: raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")
    except Exception as e: logger.error(f"PDF processing error (PyMuPDF): {e}", exc_info=True); raise HTTPException(status_code=500, detail="Server error during PDF processing.")
    finally:
        if doc: doc.close(); logger.debug("PyMuPDF doc closed.")

# --------------------------------------------------------------------------
# Data Analysis Function - *** UPDATED KEYS ***
# --------------------------------------------------------------------------
def analyze_marks(df: pd.DataFrame):
    """
    Analyzes the DataFrame, calculates stats, and returns a dictionary
    with keys matching the frontend expectation (`top_students`, `all_students`).
    """
    if df.empty:
        logger.error("Analysis error: Input DataFrame is empty.")
        raise ValueError("Cannot analyze empty dataset.")

    logger.info(f"Starting marks analysis for {len(df)} records.")
    # Log DataFrame info before cleaning
    logger.debug(f"DataFrame before cleaning: {len(df)} rows. Head:\n{df.head().to_string()}")
    try:
        THEORY_PASS_THRESHOLD = 10
        PRACTICAL_PASS_THRESHOLD = 10

        if 'Theory' not in df.columns or 'Practical' not in df.columns: raise KeyError("Missing 'Theory' or 'Practical' columns.")
        df['Theory'] = pd.to_numeric(df['Theory'], errors='coerce')
        df['Practical'] = pd.to_numeric(df['Practical'], errors='coerce')

        initial_rows = len(df)
        df = df.dropna(subset=['Theory', 'Practical'])
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0: logger.warning(f"Dropped {dropped_rows} rows due to non-numeric marks.")

        # Log DataFrame info *after* cleaning
        logger.debug(f"DataFrame after cleaning: {len(df)} rows. Head:\n{df.head().to_string()}")

        if df.empty: raise ValueError("No valid numeric marks data remained after cleaning.")

        df['Theory_Pass'] = df['Theory'] >= THEORY_PASS_THRESHOLD
        df['Practical_Pass'] = df['Practical'] >= PRACTICAL_PASS_THRESHOLD
        df['Overall_Pass'] = df['Theory_Pass'] & df['Practical_Pass']
        df['Total'] = df['Theory'] + df['Practical']
        df['Theory_Eligible'] = df['Theory_Pass'].map({True: 'Yes', False: 'No'})
        df['Practical_Eligible'] = df['Practical_Pass'].map({True: 'Yes', False: 'No'})
        df['Overall_Eligible'] = df['Overall_Pass'].map({True: 'Yes', False: 'No'})

        total_students = len(df)
        passed_count = int(df['Overall_Pass'].sum())
        failed_count = total_students - passed_count
        pass_percentage = round((passed_count / total_students) * 100, 2) if total_students > 0 else 0.0
        theory_passed_count = int(df['Theory_Pass'].sum())
        practical_passed_count = int(df['Practical_Pass'].sum())

        theory_stats = { 'average': round(df['Theory'].mean(), 2), 'max': df['Theory'].max().item(), 'min': df['Theory'].min().item(), 'passed': theory_passed_count, 'pass_percentage': round((theory_passed_count / total_students) * 100, 2) if total_students > 0 else 0.0 }
        practical_stats = { 'average': round(df['Practical'].mean(), 2), 'max': df['Practical'].max().item(), 'min': df['Practical'].min().item(), 'passed': practical_passed_count, 'pass_percentage': round((practical_passed_count / total_students) * 100, 2) if total_students > 0 else 0.0 }

        # Define columns needed for ALL tables
        # Ensure these match the columns generated above
        student_list_cols = ['SL_NO', 'USN', 'Theory', 'Theory_Eligible', 'Practical', 'Practical_Eligible', 'Total', 'Overall_Eligible']
        cols_to_select = [col for col in student_list_cols if col in df.columns]
        logger.debug(f"Columns selected for student lists: {cols_to_select}")


        # *** USE FRONTEND-EXPECTED KEYS ***
        summary = {
            'processing_summary': { 'total_records_processed': total_students, 'pass_threshold_theory': THEORY_PASS_THRESHOLD, 'pass_threshold_practical': PRACTICAL_PASS_THRESHOLD, },
            'overall_summary': { 'passed_count': passed_count, 'failed_count': failed_count, 'pass_percentage': pass_percentage, },
            'theory_stats': theory_stats,
            'practical_stats': practical_stats,
            'top_students': df.nlargest(5, 'Total')[cols_to_select].to_dict('records'), # Key changed
            'bottom_students': df.nsmallest(5, 'Total')[cols_to_select].to_dict('records'), # Key changed
            'all_students': df[cols_to_select].to_dict('records') # Key changed (was failed_students_list before)
        }
        # Log the length of lists being generated
        logger.debug(f"Generated list lengths: top={len(summary['top_students'])}, bottom={len(summary['bottom_students'])}, all={len(summary['all_students'])}")

        logger.info("Marks analysis completed successfully.")
        return summary

    except KeyError as ke: logger.error(f"Analysis KeyError: {ke}", exc_info=True); raise ValueError(f"Internal analysis error: Missing column '{ke}'.")
    except ValueError as ve: logger.error(f"Analysis ValueError: {ve}", exc_info=True); raise
    except Exception as e: logger.error(f"Analysis Unexpected Error: {e}", exc_info=True); raise ValueError(f"Unexpected analysis error.")

# --------------------------------------------------------------------------
# JSON Serializer Helper (Remains the same)
# --------------------------------------------------------------------------
def default_serializer(obj):
    # ... (serializer code is identical) ...
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)): return obj.tolist()
    elif pd.isna(obj): return None
    return str(obj)

# --------------------------------------------------------------------------
# API Endpoints (Remains the same, calls updated analyze_marks)
# --------------------------------------------------------------------------
@app.post("/summarize-marks/", tags=["Marks Processing"])
async def summarize_marks(file: UploadFile = File(..., description="PDF marks file.")):
    # ... (endpoint code is identical, calls correct functions) ...
    if not file.filename.lower().endswith('.pdf'): raise HTTPException(status_code=400, detail="Invalid file type.")
    logger.info(f"Received file: {file.filename}")
    pdf_content = None
    try:
        pdf_content = await file.read()
        if not pdf_content: raise HTTPException(status_code=400, detail="Uploaded file empty.")
        MAX_FILE_SIZE = 20 * 1024 * 1024
        if len(pdf_content) > MAX_FILE_SIZE: raise HTTPException(status_code=413, detail=f"File size exceeds limit.")
        df_marks = extract_marks_fitz(pdf_content) # Using fitz extractor
        analysis_result_dict = analyze_marks(df_marks) # Using analyzer with corrected keys
        json_compatible_string = json.dumps(analysis_result_dict, default=default_serializer)
        return JSONResponse(content=json.loads(json_compatible_string))
    except (ValueError, KeyError) as processing_err: logger.error(f"Processing Error: {processing_err}"); raise HTTPException(status_code=400, detail=f"Error processing PDF: {processing_err}")
    except HTTPException as he: raise he
    except Exception as e: logger.error(f"Endpoint Error: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Server error.")
    finally:
        if file: await file.close()
        pdf_content = None

@app.get("/", tags=["General"])
async def root(): return {"message": "Marks Summarizer API (PyMuPDF). Use /docs."}

@app.get("/health", tags=["General"])
async def health_check(): return {"status": "healthy"}

# Run: uvicorn main:app --reload --log-level debug