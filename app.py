import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from io import BytesIO
from datetime import datetime, timedelta
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["general"]["OPENAI_API_KEY"]

# Initialize the chat model
chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# Define the response schema
response_schemas = [
    ResponseSchema(name="date_due", description="The date by which the loan must be repaid"),
    ResponseSchema(name="loan_period", description="The duration of the loan in years"),
    ResponseSchema(name="loan_interest_rate", description="The annual interest rate of the loan as a decimal (e.g., 0.05 for 5%)"),
    ResponseSchema(name="gross_loan_amount", description="The gross loan amount requested"),
    ResponseSchema(name="property_value", description="The value of the property"),
    ResponseSchema(name="loan_purpose", description="The purpose of the loan (e.g., refinance, cash-out, acquisition, construction, other)"),
    ResponseSchema(name="date_purchased", description="The date the property was purchased"),
    ResponseSchema(name="purchase_price", description="The purchase price of the property"),
    ResponseSchema(name="down_payment", description="The amount of the down payment"),
    ResponseSchema(name="borrowing_entity", description="The entity borrowing the money"),
    ResponseSchema(name="borrowing_entity_type", description="The type of borrowing entity (e.g., individual, partnership, corporation, LLC, other)"),
    ResponseSchema(name="assets", description="The total assets of the borrowing entity"),
    ResponseSchema(name="liabilities", description="The total liabilities of the borrowing entity"),
    ResponseSchema(name="net_worth", description="The net worth of the borrowing entity"),
    ResponseSchema(name="total_project_completion_value", description="The total completion value of the project"),
    ResponseSchema(name="ltv_of_completed_value", description="The loan-to-value ratio of the completed value"),
    ResponseSchema(name="ltv_as_is", description="The loan-to-value ratio of the 'as is' value"),
    ResponseSchema(name="loan_to_cost", description="The loan-to-cost ratio"),
    ResponseSchema(name="total_cost_of_the_project", description="The total cost of the project"),
]

# Initialize the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Define the prompt
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("Given a loan application document, extract relevant information \n \
                                                    {format_instructions}\n{user_prompt}")
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

# Streamlit app
st.set_page_config(page_title="Bank Loan Application Analyzer", page_icon=":bank:", layout="wide")

# Create columns for the header
col1, col2 = st.columns([1, 4])

# Add company logo to the first column
with col1:
    st.image("Untitled.png", use_column_width=True)

# Add title to the second column
with col2:
    st.title("Bank Loan Application Analyzer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

def parse_loan_terms(loan_period, loan_interest_rate):
    years = int(loan_period.split()[0])
    interest_rate = float(loan_interest_rate)
    return years, interest_rate

def clean_currency(value):
    return float(''.join(filter(str.isdigit, value)))

def generate_amortization_schedule(date_due, loan_period, loan_interest_rate, gross_loan_amount):
    gross_loan_amount = clean_currency(gross_loan_amount)
    loan_years, interest_rate = parse_loan_terms(loan_period, loan_interest_rate)
    date_due = datetime.strptime(date_due, '%m/%d/%Y')
    monthly_interest_rate = interest_rate / 12
    num_payments = loan_years * 12
    monthly_payment = (monthly_interest_rate * gross_loan_amount) / (1 - (1 + monthly_interest_rate)**(-num_payments))

    dates = []
    principal_payments = []
    interest_payments = []
    remaining_balances = []
    loan_balance = gross_loan_amount

    for month in range(num_payments):
        interest_payment = loan_balance * monthly_interest_rate
        principal_payment = monthly_payment - interest_payment
        loan_balance -= principal_payment
        dates.append(date_due)
        principal_payments.append(principal_payment)
        interest_payments.append(interest_payment)
        remaining_balances.append(loan_balance)
        date_due += timedelta(days=30)

    schedule_data = {
        'Date Due': dates,
        'Principal Payment': principal_payments,
        'Interest Payment': interest_payments,
        'Remaining Balance': remaining_balances
    }
    amortization_schedule = pd.DataFrame(schedule_data)
    return amortization_schedule

if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    lines = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        lines.append(page_text.strip())
    text = '\n'.join(lines)

    user = text
    formatted_prompt = prompt.format_prompt(user_prompt=user)
    query = prompt.format_prompt(user_prompt=user)
    _output = chat_model(query.to_messages())
    output = output_parser.parse(_output.content)

    date_due = output['date_due']
    loan_period = output['loan_period']
    loan_interest_rate = output['loan_interest_rate']
    gross_loan_amount = output['gross_loan_amount']

    schedule = generate_amortization_schedule(date_due, loan_period, loan_interest_rate, gross_loan_amount)
    df_schedule = pd.DataFrame(schedule)

    # Define the loan details
    extra_values = {
        "date_due": "01/15/2030",
        "loan_period": "10 years",
        "loan_interest_rate": "0.05",
        "gross_loan_amount": "$10,000,000",
        "property_value": "$15,000,000"
    }

   

    

    # st.set_page_config(page_title='My App', layout='wide')

    # Define styles
    styles = {
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"},
        "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#4CAF50", "color": "white", "font-weight": "bold"},
    }

    # Create option menu in the sidebar
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "CSV", "Output"],
            icons=["house", "file-csv", "file-text"],
            default_index=0,
            orientation="vertical",
            styles=styles
        )


    if selected == "Dashboard":
        
        # Display the title
        # st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Amortization Schedule</h1>", unsafe_allow_html=True)

        # # Display your KPIs
        # for key, value in extra_values.items():
        #     st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{key}: {value}</h3>", unsafe_allow_html=True)
        #         # Create interactive plots
        st.markdown("""
                    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                    <div style="flex-basis: 40%; margin: 20px;">
                        <div style="font-size: 16px; font-weight: bold;">Loan End Date</div>
                        <div style="font-size: 32px; color: #8CC63E; font-weight: bold;">01/15/2030</div>
                    </div>
                    <div style="flex-basis: 40%; margin: 20px;">
                        <div style="font-size: 16px; font-weight: bold;">Loan Period</div>
                        <div style="font-size: 32px; color: #8CC63E; font-weight: bold;">10 years</div>
                    </div>
                    <div style="flex-basis: 40%; margin: 20px;">
                        <div style="font-size: 16px; font-weight: bold;">Total Paid</div>
                        <div style="font-size: 32px; color: #8CC63E; font-weight: bold;">$10,000,000</div>
                    </div>
                    <div style="flex-basis: 40%; margin: 20px;">
                        <div style="font-size: 16px; font-weight: bold;">Interest Only Payment</div>
                        <div style="font-size: 32px; color: #8CC63E; font-weight: bold;">$4,375.00</div>
                    </div>

                    </div>
"""
                    , unsafe_allow_html=True)
        
        fig = go.Figure()

        # Add Principal Payment trace
        fig.add_trace(go.Scatter(
            x=df_schedule['Date Due'],
            y=df_schedule['Principal Payment'],
            mode='lines+markers',
            name='Principal Payment'
        ))

        # Add Interest Payment trace
        fig.add_trace(go.Scatter(
            x=df_schedule['Date Due'],
            y=df_schedule['Interest Payment'],
            mode='lines+markers',
            name='Interest Payment'
        ))

        # Add Remaining Balance trace on secondary y-axis
        fig.add_trace(go.Scatter(
            x=df_schedule['Date Due'],
            y=df_schedule['Remaining Balance'],
            mode='lines+markers',
            name='Remaining Balance',
            yaxis='y2'
        ))

        # Update layout
        fig.update_layout(
            title='Amortization Schedule',
            xaxis_title='Date Due',
            yaxis_title='Amount ($)',
            legend_title='Legend',
            hovermode='x unified',
            yaxis2=dict(
                title='Remaining Balance ($)',
                overlaying='y',
                side='right'
            ),

            width=1500,  # Set the width of the plot
            height=800 ,
  
        )

        # Display the plot

        st.plotly_chart(fig)

    elif selected == "CSV":
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Amortization Schedule CSV</h1>", unsafe_allow_html=True)
        st.write("Amortization Schedule:")
        st.dataframe(df_schedule)

        # Provide download option for the CSV
        csv = df_schedule.to_csv(index=False)
        csv_buffer = BytesIO()
        csv_buffer.write(csv.encode('utf-8'))
        csv_buffer.seek(0)
        st.download_button(
            label="Download Amortization Schedule as CSV",
            data=csv_buffer,
            file_name='amortization_schedule.csv',
            mime='text/csv'
        )

    elif selected == "Output":
        st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Loan Application Output</h1>", unsafe_allow_html=True)
        st.json(output)
