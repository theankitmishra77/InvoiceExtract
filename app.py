from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import base64
import ssl
import re
import os
from io import BytesIO
import time
import json
from time import sleep
from datetime import date
from datetime import datetime
from textractor import Textractor
from dateutil import parser
from datetime import datetime
#from langchain.chat_models import ChatOpenAI 
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from textractor.data.constants import TextractFeatures
from textractor.data.constants import AnalyzeExpenseFields, AnalyzeExpenseFieldsGroup, AnalyzeExpenseLineItemFields
from textractcaller.t_call import Textract_Features, Textract_Types, call_textract, call_textract_expense
from textractprettyprinter.t_pretty_print_expense import get_expenselineitemgroups_string,get_expensesummary_string,convert_expensesummary_to_list, convert_expenselineitemgroup_to_list
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import os
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "credentials"
os.environ["AWS_CONFIG_FILE"] = "config"

from textractor import Textractor
extractor = Textractor(profile_name="ankit")


from textractcaller.t_call import call_textract, Textract_Features
import pandas as pd
load_dotenv()

# Access the OpenAI API key from the environment
#openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the API key is loaded
# if not openai_api_key:
#     raise ValueError("OpenAI API key not found. Ensure it is set in the .env file.")

# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

#os.environ["OPENAI_API_KEY"] = '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' 

extractor = Textractor(profile_name="ankit")

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

def expense_output(path):
    document = extractor.analyze_expense(
        file_source=r"{}".format(path),
        save_image=True,
    )
    dc = document
    expense_doc = dc.expense_documents[0]
    return expense_doc
	
def get_header_details(expense_doc, pdf_name, doc_number, company_code):
    Dic_header = {}
    Dic_header['COMPANY_CODE'] = Delivery_Date
    Dic_header['DOC_NUMBER'] = doc_number
    Dic_header['PDF_NAME'] = pdf_name
    return Dic_header
    
def convert_pdf_to_images(pdf_path):
    """
    Converts a PDF to images and saves them in the same folder with the same file name.
    Each page of the PDF will be saved as a separate image.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    # Get the directory and file name without the extension
    #directory, file_name = os.path.split(pdf_path)
    #base_name, _ = os.path.splitext(file_name)

    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=300)  # Adjust DPI for quality
        for i, image in enumerate(images):
            # Save each page as an image in the same folder
            output_path = os.path.join(os.getcwd(), f"{pdf_path.split('.')[0]}.png")
            image.save(output_path, "PNG")
            print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error converting {pdf_path} to images: {e}")
        
def extract_key_value_pairs(text_list):
    """
    Extracts key-value pairs from a list of strings where keys are inside brackets
    and values are after a colon (:).

    Args:
        text_list (list): A list of strings containing key-value pairs.

    Returns:
        dict: A dictionary containing the extracted key-value pairs.
    """
    pattern = re.compile(r"\((.*?)\):\s*(.+)")
    extracted_dict = {}

    for text in text_list:
        matches = pattern.findall(str(text))
        for key, value in matches:
            extracted_dict[key.strip()] = value.strip()

    return extracted_dict
        
        
@app.route('/extract-pdf-data', methods=['POST'])
def extract_pdf():
    header_keys = [
        "PO_NO",
        "INVOICE_DATE",
        "POSTING_DATE",
        "REFERENCE",
        "AMOUNT",
        "TAX_AMOUNT",
	"CGST_TAX_AMOUNT",
	"SGST_TAX_AMOUNT",
	"IGST_TAX_AMOUNT",
        "DOC_CURRENCY",
        "VENDOR",
        "VENDOR_NAME",
        "VENDOR_STATE",
        "BUYER_GSTIN",
        "VENDOR_GSTIN",
        "BUSINESS_PLACE",
        "SECTION_CODE",
        "REVERSE_CHARGE",
        "PAYMENT_TERM",
        "BASE_LINE_DATE",
        "TOTAL_INVOICE_AMOUNT",
        "DUE_DATE",
        "WITH_HOLDING_TAX",
    ]

    item_keys = [
        'DOC_NUMBER',
        'ITEM_NO',
        'MATNR',
        'MATERIAL_DESCRIPTION',
        'QUANTITY',
        'RATE',
        'UOM',
        'AMOUNT',
        'PLANT',
        'TAX_PERCENT',
	"CGST_RATE",
	"SGST_RATE",
	"IGST_RATE",
        'TAX_CODE',
        'HSN_CODE',
        'COST_CENTER',
        'PROFIT_CENTER',
        'REFERENCE_DOC',
        'GOODS_RECEIPT_QTY',
        'PREVIOUS_INV_QTY',
        'PO_TOT_QTY'
    ]
    try:
        # Parse and validate the request payload
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
        # Save the decoded PDF content to a file
        try:
            with open(pdf_name, 'wb') as pdf_file:
                pdf_file.write(pdf_content)
        except IOError as e:
            return jsonify({"error": f"Failed to save PDF file: {str(e)}"}), 500
        print('3')
        # Process the PDF to extract data
        expense_doc = expense_output(pdf_name)
 #       pdf_file_path = r"D:\Tejas\Vadilal\MAV246409.pdf"  # Replace with your PDF path
        convert_pdf_to_images(pdf_name)
        print(expense_doc)
        documentName = f"{pdf_name.split('.')[0]}.png"
                
        j = call_textract_expense(input_document=documentName)
        Complete = get_expensesummary_string(textract_json = j)

        prompt_template = """
        Extract the following fields from the given text using contextual understanding:
        {fields}

        Text:
        {text}

        Notes:
        1. Fields must be filled where information can be interpreted logically or explicitly mentioned.
        2. For fields like State, interpret the location context (e.g., 'State: Gujarat' implies 'Gujarat').
        3. For currency, infer from the monetary values or standard assumptions (e.g., amounts in 'INR').
        4. Fields that are not explicitly or implicitly clear should be left empty.
        5. Fields like BUISNESS_PLACE, SECTION_CODE, VENDOR(this is not GST No.) are unique Codes so mult be identified carefully or to be left empty. 
        6. Fields that can be converted to integer or float should be parsed as integer and float respectively.
        7. Date fields should have dates formatted as 'YYYY-MM-DD'.
	8. PO Number and Buyers Order No. are exactly same. 
        9. REFERENCE is the Invoice No. so capture it carefully without any error.
	10. All the tax amounts should be captured carefully.

        Output the extracted fields in JSON format, preserving the field order.
        """

        # Create the LangChain prompt
        prompt = PromptTemplate(
            input_variables=["fields", "text"],
            template=prompt_template,
        )

        # Initialize the OpenAI LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Adjust model if needed

        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Run the chain with the input text and fields
        header = chain.run(fields=", ".join(header_keys), text=Complete)

        # Output the result
        print(header)
        
        data_dicts = []
        Final = []


        for j in expense_doc.line_items_groups[0]:
            Empty = []
            for i in j:
                print(i)
                print('=='*30)
                Empty.append(str(i).replace('\n', ' ').replace('\\n',''))
            Final.append(Empty)
            
        for i in Final:
            result = extract_key_value_pairs(i)
            #print(result)
            data_dicts.append(result)
        # for i in expense_doc.line_items_groups[0]:
        #     result = extract_key_value_pairs(i._line_item_expense_fields)
        #     print(result)
        #     data_dicts.append(result)
            
        Items = pd.DataFrame(data_dicts)
        Items

        # Prompt Template
        prompt_template = """
        Extract the following fields for each item in the given dictionary:
        {fields}

        Dictionary:
        {items}

        Notes:
	1. 'ITEM_NO' field is a serial no. starting from 1 and goes on increasing according to no. of items.
        2. Use the dictionary keys to derive values where possible.
        3. If a field is not available, leave it empty.
        4. Format the output as a JSON list with objects for each item.
        5. Fields that can be converted to integer or float should be parsed as integer and float respectively.
        6. Reference No is same as Invoice Number.
        7. Date fields should have dates formatted as 'YYYY-MM-DD'.
	8. All the rates and rates(%) should be captured and parsed carefully.

        Output:
        """
        # Create the LangChain prompt
        prompt = PromptTemplate(
            input_variables=["fields", "items"],
            template=prompt_template,
        )

        # Initialize the OpenAI LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Adjust model if needed

        # Create the LangChain chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Prepare input for LangChain
        items_input = str(Items.to_dict())

        # Run the chain
        items = chain.run(fields=", ".join(item_keys), items=items_input)

        # Print the result
        print(items)
        header_data = json.loads(header)
        items_data = json.loads(items)
        response = {
            "header": header_data,
            "items": items_data,
        }
        response['header']['DOC_NUMBER'] = data['DOC_NUMBER']
        response['header']['COMPANY_CODE'] = data['COMPANY_CODE']
        response['header']['PDF_NAME'] = data['PDF_NAME']
        response['header']['MSG_STATUS'] = 'S'
        response['header']['ERROR_MSG'] = ''
        print('Final Response',response)
        return jsonify(response)
    except Exception as e:
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
