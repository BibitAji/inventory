import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# Path to your credentials file
credentials_file = "entry-form-423700-59d9ca132408.json"

# Define the scope
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

# Authenticate using the credentials file
creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
client = gspread.authorize(creds)

# Open the Google Sheet using the URL provided
sheet_url = "https://docs.google.com/spreadsheets/d/1cwZZ-TbqnpLAHLRaDIoAGSzZa5aPKn4u9Ni91sj9IAY/edit?usp=sharing"
sheet = client.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)

def add_row(data):
    worksheet.append_row(data)
    return f"Row with data '{data}' has been added."

def delete_row_by_part_number(part_number):
    records = worksheet.get_all_records()
    for idx, record in enumerate(records, start=2):  # start=2 to account for the header row
        if record.get("Part Number") == part_number:
            worksheet.delete_rows(idx)
            return f"Row with Part Number '{part_number}' has been deleted."
    return f"No record found with the Part Number '{part_number}'."

def update_component(part_number, new_component):
    records = worksheet.get_all_records()
    for idx, record in enumerate(records, start=2):  # start=2 to account for the header row
        if record.get("Part Number") == part_number:
            worksheet.update_cell(idx, 2, new_component)  # Assuming "Component" is the second column
            return f"Component for Part Number '{part_number}' has been updated to '{new_component}'."
    return f"No record found with the Part Number '{part_number}'."

def main():
    st.title("Google Sheets Data Management | Bibit Waluyo Aji")

    # Read Google Sheets and display DataFrame
    tabel_csv = sheet_url.replace('/edit?usp=sharing','/gviz/tq?tqx=out:csv' )
    tabel_df = pd.read_csv(tabel_csv)
    st.write(tabel_df)

    # Search bar
    search = st.text_input("Search in Data")
    if search:
        df1 = tabel_df[tabel_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
        st.write(df1)  

    # Data Entry Form
    st.header("Add Data to Sheet")
    with st.form(key='add_data_form'):
        part_number = st.text_input('Part Number')
        component = st.text_input('Component')
        quantity = st.text_input('Quantity')
        annual_demand = st.text_input('Annual Demand')
        std_deviation = st.text_input('Std. Deviation')
        lead_time = st.text_input('Lead Time')
        unit_cost = st.text_input('Unit Cost')
        ordering_cost = st.text_input('Ordering Cost')
        storage_cost = st.text_input('Storage Cost')
        shortage_cost = st.text_input('Shortage Cost')
        add_button = st.form_submit_button(label='Add')

    if add_button:
        new_row = [
            part_number, component, quantity, annual_demand, std_deviation,
            lead_time, unit_cost, ordering_cost, storage_cost, shortage_cost
        ]
        result = add_row(new_row)
        st.write(result)

    # Data Deletion Form
    st.header("Delete Data from Sheet")
    with st.form(key='delete_data_form'):
        part_number_to_delete = st.text_input('Part Number to Delete')
        delete_button = st.form_submit_button(label='Delete')

    if delete_button:
        if not part_number_to_delete:
            st.error("Please provide a Part Number to delete.")
        else:
            result = delete_row_by_part_number(part_number_to_delete)
            st.write(result)
            # Refresh the data
            tabel_df = pd.read_csv(tabel_csv)
            st.write(tabel_df)

    # Data Update Form
    st.header("Update Component in Sheet")
    with st.form(key='update_data_form'):
        part_number_to_update = st.text_input('Part Number to Update')
        new_component = st.text_input('New Component')
        update_button = st.form_submit_button(label='Update')

    if update_button:
        if not part_number_to_update or not new_component:
            st.error("Please provide both a Part Number and a new Component.")
        else:
            result = update_component(part_number_to_update, new_component)
            st.write(result)
            # Refresh the data
            tabel_df = pd.read_csv(tabel_csv)
            st.write(tabel_df)

if __name__ == '__main__':
    main()
