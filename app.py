import base64
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import LLMChain
from anthropic import Anthropic
import logging
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API keys
#ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_API_KEY= os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Validate API keys
if not ANTHROPIC_API_KEY or not OPENAI_API_KEY:
    raise EnvironmentError("Missing required API keys. Ensure ANTHROPIC_API_KEY and OPENAI_API_KEY are set in .env")

# Initialize clients
client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Define prompt template for transformation
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Transform the following JSON data into the required JSON (key and value) format with both `header` and `items` keys understanding contextual fields of `header` and `items`:

Important Notes:
1. 'REFERENCE' is same as Invoice Number.
2. 'TAX_AMOUNT' =  'CGST_TAX_AMOUNT'+'SGST_TAX_AMOUNT'+IGST_TAX_AMOUNT'
3. 'TOTAL_INVOICE_AMOUNT' = 'AMOUNT'+'TAX_AMOUNT'
4. 'IGST_RATE','CGST_RATE','SGST_RATE' if applicable to the entire PDF should be applicable to each item. 
5. 'VENDOR' is a unique identifier for Vendor and if not found should be left empty.
6. Date fields should have dates formatted as 'YYYY-MM-DD'
7. 'ITEM_NO' should start from 1 and increase consecutively.
8. 'HSN CODE' is same as 'SAC' or 'HSN/SAC'
9. 'UOM' represents Unit of Measurement so extract it for each item, remember it would be same for each item.
10. 'INVOICE_TYPE' and 'PO_NUMBER' should be captured accurately.
11. 'DOC_CURRENCY' is Document Currency (e.g., INR for Indian Rupees, USD for US Dollars, etc.).

CRITICAL TAX DETERMINATION RULES:
  1. IGST rate if 0 then Only CGST and SGST have equal non-zero values.
  2. IGST rate if not 0 then CGST and SGST both are 0.
  3. If seller and buyer states are different:
        - IGST rate must be the total tax rate (18% if shown)
        - CGST and SGST rates must both be exactly 0%
        - CGST and SGST amounts must both be exactly 0.00
  4. If seller and buyer states are same:
        - IGST rate and amount must be exactly 0
        - CGST and SGST rates must each be half of total rate (9% each if total is 18%)
        - CGST and SGST amounts must be equal

Input:
{text}

Output format:
{{
    'header': {{
        'PO_NO': '',
        'INVOICE_TYPE':'',
        'INVOICE_DATE': '',
        'INVOICE_NO': '',
        'AMOUNT': '',
        'TAX_AMOUNT': '',
        'CGST_TAX_AMOUNT': '',
        'SGST_TAX_AMOUNT': '',
        'IGST_TAX_AMOUNT': '',
        'DOC_CURRENCY': '',
        'VENDOR_NAME': '',
        'VENDOR_STATE': '',
        'BUYER_GSTIN': '',
        'VENDOR_GSTIN': '',
        'REVERSE_CHARGE': '',
        'TOTAL_INVOICE_AMOUNT': '',
        'PDF_NAME': '',
        'MSG_STATUS': '',
        'ERROR_MSG': ''
        'EMAIL':''
    }},
    'items': [
        {{
            'PO_NO': '',
            'ITEM_NO': '',
            'MATERIAL_DESCRIPTION': '',
            'QUANTITY': '',
            'RATE': '',
            'UOM': '',
            'AMOUNT': '',
            'TAX_PERCENT': '',
            'CGST_RATE': '',
            'SGST_RATE': '',
            'IGST_RATE': '',
            'HSN_CODE': '',
        }}
    ]
}}
All numerical field should be of type decimal having 2 decimal places.Also 'PO_NO' value should be any of PO Number, Customer PO , PO#, P.O#.

"""
)

def is_email(string):
    # Define the regex pattern for a valid email
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Match the string against the regex
    return re.match(email_regex, string) is not None

# Function to process PDF and extract invoice data
def extract_invoice_data(pdf_path): 
    try:
        # Convert PDF to image
        images = convert_from_path(pdf_path)
        img_byte_arr = BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Anthropic call to extract raw JSON
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all possible information from this invoice PDF and provide the output in JSON format. Ensure the extraction includes, but is not limited to, the following details with absolute precision:

                                Document Details:
                                - PO Number or Customer PO, PO#, P.O#, Customer Reference, Customer Ref No., Customer Reference Number.
                                - Invoice Number, INV NO, INV#, INVOICE NO, Invoice Reference Number, Reference Number, Invoice Ref No..
                                - Document Currency (e.g., INR for Indian Rupees, USD for US Dollars, etc.)
                                - Invoice Type -> ("Standard Invoice",
                                            "Proforma Invoice",
                                            "Commercial Invoice",
                                            "Credit Invoice (Credit Memo)",
                                            "Debit Invoice (Debit Memo)",
                                            "Timesheet Invoice",
                                            "Recurring Invoice",
                                            "Interim Invoice",
                                            "Final Invoice",
                                            "Past Due Invoice",
                                            "E-Invoice (Electronic Invoice)",
                                            "Utility Invoice",
                                            "Expense Invoice",
                                            "Tax Invoice",
                                            "Retainer Invoice")
                                - Invoice and Due Dates
                                
                                Seller and Buyer Details:
                                - Seller Name, Address, Vendor GSTIN (very accurately)
                                - Buyer Name, Address, Buyer GSTIN (very accurately)
                                
                                Line Items: 
                                - Item Description
                                - Quantity
                                - HSN COde or SAC Code
                                - Unit of Measurement (UOM)
                                - Unit Price or Rate
                                - Total Amount (ensure precision in calculation)
                                
                                Amounts and Totals:
                                - Subtotal Amount
                                - Tax Amount (if applicable, with type and percentage)
                                - Total Amount (including taxes and discounts, if any)
                                - Discounts (if applicable)
                                
                                Reverse Charge:  
                                - Is the tax payable on a reverse charge basis? (Yes/No)  
                                
                                Requirements:
                                - Ensure all numeric details, such as rates, amounts, and totals, are extracted accurately without errors.
                                - Map document currency (DOC_CURRENCY) intelligently based on the symbols or abbreviations present in the document (e.g., INR for ₹, USD for $).
                                - Include all dates in a consistent format (e.g., YYYY-MM-DD).
                                - Capture all data fields, even if they appear in unconventional formats or positions within the document."""
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        )

        # Extracting the response text
        response_text = message.content[0].text

        # Debugging the response
        print("Raw Response Text:", response_text)

        # Clean and validate JSON format
        if response_text.startswith('```json'):
            response_text = response_text.split('```json', 1)[-1]
        if response_text.endswith('```'):
            response_text = response_text[:-3]

        # Validate JSON before decoding
        response_text = response_text.strip()
        try:
            invoice_data = json.loads(response_text)
        except json.JSONDecodeError as decode_err:
            logging.error(f"JSON Decode Error: {decode_err}")
            raise ValueError(f"Invalid JSON format in response: {response_text}")

        print("Extracted Invoice Data:", invoice_data)
        return invoice_data
    except Exception as e:
        logging.error(f"Error extracting invoice data: {e}")
        return {"error": str(e)}

# Function to transform invoice data
def transform_invoice_data(invoice_data):
    try:
        # Verify the invoice data has the expected structure
        if not isinstance(invoice_data, dict):
            raise ValueError("Input data for transformation must be a dictionary.")

        # Convert the invoice data to JSON string for LangChain
        input_text = json.dumps(invoice_data, indent=2)
        print("Prepared Input for LangChain:", input_text)
        
        # Log the prepared input
        logging.info("Prepared input for LangChain transformation.")
        
        # Create the chain (assuming `chain` and `prompt_template` are defined elsewhere)
        chain = LLMChain(llm=openai_client, prompt=prompt_template)

        # Run the transformation
        header = chain.run(text=input_text)
        print("Transformation Output:", header)
        header = header.replace("'", '"')
        parsed_header = json.loads(header)
        print("Parsed Header:", parsed_header)
        print(type(parsed_header))
        # Format the JSON for pretty output
        formatted_header = json.dumps(parsed_header, indent=2)
        print(type(formatted_header))
        return parsed_header
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in transformation: {e}")
        return {"error": f"Failed to decode JSON: {e}"}
    except ValueError as e:
        logging.error(f"Value error in transformation: {e}")
        return {"error": str(e)}
    except Exception as e:
        logging.error(f"Unexpected error during transformation: {e}")
        return {"error": f"Unexpected error: {e}"}


# Flask API endpoint
@app.route('/process-invoice', methods=['POST'])
def process_invoice():
    try:
        data = request.get_json()
        if not data or 'DOC_NUMBER' not in data or 'COMPANY_CODE' not in data or 'PDF_NAME' not in data or 'PDF_CONTENT' not in data:
            return jsonify({"error": "Missing required fields: 'DOC_NUMBER', 'COMPANY_CODE', 'PDF_NAME', or 'PDF_CONTENT'"}), 400

        # Extract and validate payload data
        doc_number = data['DOC_NUMBER']
        company_code = data['COMPANY_CODE']
        pdf_name = data['PDF_NAME']
        pdf_content_encoded = data['PDF_CONTENT']

        # Decode Base64 content
        print('1')
        try:
            pdf_content = base64.b64decode(pdf_content_encoded)
        except base64.binascii.Error as e:
            #return jsonify({"error": f"Base64 decoding failed: {str(e)}"}), 400
            return jsonify({
            "header": {
                "MSG_STATUS": "E",
                "ERROR_MSG": f"Base64 decoding failed: {str(e)}",
                "ERROR_NO": 504,
                "PDF_NAME": data['PDF_NAME']
            },
            "items": []
        }), 500
        print(pdf_name)        # Ensure the PDF name has a .pdf extension
        if not pdf_name.endswith(".pdf"):
            pdf_name += ".pdf"
        print('2')
        try:
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(pdf_content)
        except IOError as e:
            #return jsonify({"error": f"Failed to save PDF file: {str(e)}"}), 500
            return jsonify({
            "header": {
                "MSG_STATUS": "E",
                "ERROR_MSG": f"Failed to save PDF file: {str(e)}",
                "ERROR_NO": 505,
                "PDF_NAME": data['PDF_NAME']
            },
            "items": []
        }), 500
        # Extract and transform invoice data
        raw_data = extract_invoice_data(pdf_name)
        if "error" in raw_data:
            #return jsonify({"error": "Failed to extract data from invoice."}), 500
            return jsonify({
            "header": {
                "MSG_STATUS": "E",
                "ERROR_MSG": "Failed to extract data from invoice.",
                "ERROR_NO": 506,
                "PDF_NAME": data['PDF_NAME']
            },
            "items": []
        }), 500
        transformed_data = transform_invoice_data(raw_data)
        if "error" in transformed_data:
            #return jsonify({"error": "Failed to transform invoice data."}), 500
            return jsonify({
            "header": {
                "MSG_STATUS": "E",
                "ERROR_MSG": "Failed to transform invoice data.",
                "ERROR_NO": 507,
                "PDF_NAME": data['PDF_NAME']
            },
            "items": []
        }), 500
        transformed_data['header']['DOC_NUMBER'] = data['DOC_NUMBER']
        if transformed_data['header']['INVOICE_TYPE'].startswith('Tax'):
            transformed_data['header']['INVOICE_TYPE'] = 'I'
        elif transformed_data['header']['INVOICE_TYPE'].startswith('Credit'):
            transformed_data['header']['INVOICE_TYPE'] = 'C'
        elif transformed_data['header']['INVOICE_TYPE'].startswith('Debit'):
            transformed_data['header']['INVOICE_TYPE'] = 'D'
        transformed_data['header']['COMPANY_CODE'] = data['COMPANY_CODE']
        transformed_data['header']['PDF_NAME'] = data['PDF_NAME']
        transformed_data['header']['MSG_STATUS'] = 'S'
        transformed_data['header']['ERROR_MSG'] = ''
        transformed_data['header']['ERROR_NO'] = ''
        if is_email(transformed_data['header']['PO_NO']):           
            transformed_data['header']['EMAIL'] = transformed_data['header']['PO_NO']
            transformed_data['header']['PO_NO'] = ''
            print('sdfghj')
            for item in transformed_data['items']:
                item['PO_NO'] = ''
        else:
            transformed_data['header']['EMAIL'] = ''

        return jsonify(transformed_data), 200

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return jsonify({
            "header": {
                "MSG_STATUS": "E",
                "ERROR_MSG": str(e),
                "ERROR_NO": 500,
                "PDF_NAME": data['PDF_NAME']
            },
            "items": []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
