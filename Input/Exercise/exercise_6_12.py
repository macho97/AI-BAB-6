"""Student registration form with a user-selectable database filename."""

import csv
from pathlib import Path

import streamlit as st


def main():
    st.title("Exercise 6.12 - Student Database")
    database = st.text_input("Database filename", "database.txt")
    with st.form("student"):
        student_id = st.text_input("ID")
        name = st.text_input("Name")
        course = st.text_input("Course")
        address = st.text_input("Address")
        submitted = st.form_submit_button("Save")
    if submitted:
        if not student_id or not name or not database.strip():
            st.error("ID, name, and database filename are required.")
            return
        path = Path(database.strip())
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(("Id", "Name", "Course", "Address"))
            writer.writerow((student_id, name, course, address))
        st.success(f"Saved student record in {path}")


if __name__ == "__main__":
    main()
