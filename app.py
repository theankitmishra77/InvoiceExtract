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
{
    'header': {
        'PO_NO': '',
        'INVOICE_DATE': '',
        'POSTING_DATE': '',
        'REFERENCE': '',
        'AMOUNT': '',
        'TAX_AMOUNT': '',
        'CGST_TAX_AMOUNT': '',
        'SGST_TAX_AMOUNT': '',
        'IGST_TAX_AMOUNT': '',
        'DOC_CURRENCY': '',
        'VENDOR': '',
        'VENDOR_NAME': '',
        'VENDOR_STATE': '',
        'BUYER_GSTIN': '',
        'VENDOR_GSTIN': '',
        'BUSINESS_PLACE': '',
        'SECTION_CODE': '',
        'REVERSE_CHARGE': '',
        'PAYMENT_TERM': '',
        'BASE_LINE_DATE': '',
        'TOTAL_INVOICE_AMOUNT': '',
        'DUE_DATE': '',
        'WITH_HOLDING_TAX': '',
        'DOC_NUMBER': '',
        'COMPANY_CODE': '',
        'PDF_NAME': '',
        'MSG_STATUS': '',
        'ERROR_MSG': ''
    },
    'items': [
        {
            'DOC_NUMBER': '',
            'ITEM_NO': '',
            'MATNR': '',
            'MATERIAL_DESCRIPTION': '',
            'QUANTITY': '',
            'RATE': '',
            'UOM': '',
            'AMOUNT': '',
            'PLANT': '',
            'TAX_PERCENT': '',
            'CGST_RATE': '',
            'SGST_RATE': '',
            'IGST_RATE': '',
            'TAX_CODE': '',
            'HSN_CODE': '',
            'COST_CENTER': '',
            'PROFIT_CENTER': '',
            'REFERENCE_DOC': '',
            'GOODS_RECEIPT_QTY': '',
            'PREVIOUS_INV_QTY': '',
            'PO_TOT_QTY': ''
        }
    ]
}
All numerical fields should be of type decimal having 2 decimal places.
    """
)

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
            temperature = 0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all the information from this invoice in JSON format. Include all details like invoice number, po_number, billing_document_number, dates, seller details, buyer details, items, amounts, and payment terms."
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

        
        response_text = message['completion']
        invoice_data = json.loads(response_text)

        return invoice_data
    except Exception as e:
        logging.error(f"Error extracting invoice data: {e}")
        return {"error": str(e)}

# Function to transform invoice data
def transform_invoice_data(invoice_data):
    try:
        chain = LLMChain(llm=openai_client, prompt=prompt_template)
        formatted_data = chain.run(text=json.dumps(invoice_data, indent=2))
        return json.loads(formatted_data)
    except Exception as e:
        logging.error(f"Error transforming invoice data: {e}")
        return {"error": str(e)}

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
            return jsonify({"error": f"Base64 decoding failed: {str(e)}"}), 400
        print(pdf_name)        # Ensure the PDF name has a .pdf extension
        if not pdf_name.endswith(".pdf"):
            pdf_name += ".pdf"
        print('2')
        try:
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(pdf_content)
        except IOError as e:
            return jsonify({"error": f"Failed to save PDF file: {str(e)}"}), 500

        # Extract and transform invoice data
        raw_data = extract_invoice_data(pdf_name)
        if "error" in raw_data:
            return jsonify({"error": "Failed to extract data from invoice."}), 500

        transformed_data = transform_invoice_data(raw_data)
        if "error" in transformed_data:
            return jsonify({"error": "Failed to transform invoice data."}), 500
            
        transformed_data['header']['DOC_NUMBER'] = data['DOC_NUMBER']
        transformed_data['header']['COMPANY_CODE'] = data['COMPANY_CODE']
        transformed_data['header']['PDF_NAME'] = data['PDF_NAME']
        transformed_data['header']['MSG_STATUS'] = 'S'
        transformed_data['header']['ERROR_MSG'] = ''

        return jsonify(transformed_data), 200

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
        return jsonify({
            "header": {
                "MSG_STATUS": "E",
                "ERROR_MSG": str(e),
            },
            "items": []
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
