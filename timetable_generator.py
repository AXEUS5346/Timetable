import os
import json
import re
from datetime import datetime
import pandas as pd
from datetime import datetime, timedelta
import csv
import time
import shutil
import tkinter as tk
from tkinter import filedialog

# LLM Provider options
from groq import Groq  # For Groq API
# Using updated import to fix deprecation warning
from langchain_ollama import OllamaLLM  # For Ollama local models


class CsvSaver:
    """
    Class responsible for saving CSV content to files with various options.
    This class handles all the CSV saving functionality for the timetable generator.
    """
    
    def __init__(self, default_folder="timetables"):
        """Initialize the CSV Saver with default save location"""
        self.default_folder = default_folder
        
    def get_save_location(self):
        """Get the user's preferred location to save the CSV file"""
        print("\n===== SAVE LOCATION =====")
        print(f"Default save location: {os.path.abspath(self.default_folder)}")
        
        # Ask user if they want to use default location or choose a different one
        choice = input("Save to default location? (y/n): ").lower()
        
        if choice == 'y' or choice == '':
            # Use default location, ensure it exists
            if not os.path.exists(self.default_folder):
                try:
                    os.makedirs(self.default_folder)
                    print(f"Created folder: {os.path.abspath(self.default_folder)}")
                except Exception as e:
                    print(f"Warning: Could not create default folder: {e}")
                    print("Will attempt to save in current directory instead.")
                    return ""
            return self.default_folder
        else:
            # Use file dialog to choose location
            try:
                # Initialize tkinter and hide the main window
                root = tk.Tk()
                root.withdraw()
                
                # Show directory selection dialog
                print("Please select a folder to save the timetable...")
                save_dir = filedialog.askdirectory(title="Select folder to save timetable")
                
                # Clean up tkinter
                root.destroy()
                
                if save_dir:
                    print(f"Selected save location: {save_dir}")
                    return save_dir
                else:
                    print("No folder selected. Using default location.")
                    return self.default_folder
            except Exception as e:
                print(f"Error using folder dialog: {e}")
                
                # Fallback to manual input if dialog fails
                custom_path = input("Enter full path to save folder: ")
                if custom_path and os.path.exists(custom_path):
                    return custom_path
                else:
                    if custom_path:
                        try:
                            os.makedirs(custom_path)
                            print(f"Created folder: {custom_path}")
                            return custom_path
                        except Exception as e:
                            print(f"Could not create folder: {e}")
                    
                    print("Using default location.")
                    return self.default_folder
    
    def generate_filename(self, institute_name, batch_info, start_date=None, end_date=None):
        """Generate a filename based on institute, batch, and date range"""
        # Clean institute name
        institute_name = institute_name.replace(" ", "_") if institute_name else "Institute"
        
        # Format batch information if available
        batch_suffix = ""
        if batch_info:
            if isinstance(batch_info, list) and batch_info:
                batch = batch_info[0]
            else:
                batch = batch_info
            batch_suffix = f"_{str(batch).replace(' ', '_')}"
        
        # Create filename with date range or timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        if start_date and end_date:
            # Convert dates to clean format for filename - replace any punctuation
            start_clean = str(start_date).replace("/", "-").replace(" ", "_").replace(",", "")
            end_clean = str(end_date).replace("/", "-").replace(" ", "_").replace(",", "")
            filename = f"{institute_name}{batch_suffix}_timetable_{start_clean}_to_{end_clean}.csv"
        else:
            # Fallback to timestamp
            filename = f"{institute_name}{batch_suffix}_timetable_{timestamp}.csv"
            
        return filename
    
    def save_csv(self, csv_content, institute_name="Institute", batch_info=None, 
                 start_date=None, end_date=None, custom_filename=None, save_location=None):
        """
        Save CSV content to a file.
        
        Args:
            csv_content (str): The CSV content to save
            institute_name (str): Name of the institute for filename generation
            batch_info (str or list): Batch information for filename generation
            start_date (str): Start date for filename generation
            end_date (str): End date for filename generation
            custom_filename (str): Optional custom filename to use instead of generated one
            save_location (str): Optional specific save location to use
            
        Returns:
            tuple: (success (bool), filepath (str), error_message (str or None))
        """
        # Always use the default folder without prompting
        output_folder = save_location if save_location else self.default_folder
        
        # Ensure the default folder exists
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
                print(f"Created folder: {os.path.abspath(output_folder)}")
            except Exception as e:
                print(f"Warning: Could not create default folder: {e}")
                print("Will save in current directory instead.")
                output_folder = ""
        
        # Generate or use custom filename
        if custom_filename:
            # Ensure filename ends with .csv
            if not custom_filename.lower().endswith('.csv'):
                filename = f"{custom_filename}.csv"
            else:
                filename = custom_filename
        else:
            filename = self.generate_filename(institute_name, batch_info, start_date, end_date)
        
        # Full path to save file
        filepath = os.path.join(output_folder, filename)
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Save the file
            with open(filepath, "w", newline='', encoding='utf-8') as f:
                f.write(csv_content)
            
            # Get full absolute path to display to user
            full_path = os.path.abspath(filepath)
            
            print(f"\n✅ CSV file successfully saved!")
            print(f"📄 File: {filename}")
            print(f"📁 Location: {full_path}")
            
            # Create backup copy
            self._create_backup(filepath, filename)
            
            # Preview the saved content
            self._show_preview(csv_content)
            
            # Offer to open the folder
            self._offer_open_folder(full_path)
            
            return True, full_path, None
            
        except Exception as e:
            error_message = f"Error saving file: {e}"
            print(f"\n❌ {error_message}")
            
            # Try saving to current directory as fallback
            try:
                fallback_path = os.path.join(os.getcwd(), filename)
                with open(fallback_path, "w", newline='', encoding='utf-8') as f:
                    f.write(csv_content)
                print(f"Saved to current directory instead: {fallback_path}")
                return True, fallback_path, error_message
            except Exception as e2:
                final_error = f"Failed to save file to fallback location: {e2}"
                print(final_error)
                return False, None, f"{error_message}. {final_error}"
    
    def _create_backup(self, filepath, filename):
        """Create a backup copy of the saved file"""
        try:
            backup_folder = os.path.join(os.path.dirname(filepath), "backups")
            if not os.path.exists(backup_folder):
                os.makedirs(backup_folder)
            backup_path = os.path.join(backup_folder, f"backup_{filename}")
            shutil.copy2(filepath, backup_path)
            print(f"💾 Backup copy saved to: {backup_path}")
        except Exception as e:
            # Non-critical error, just log it
            print(f"Note: Could not create backup: {e}")
    
    def _show_preview(self, csv_content, max_lines=10):
        """Show a preview of the CSV content"""
        print("\n===== SAMPLE OF SAVED CSV =====")
        preview_lines = csv_content.split('\n')[:max_lines]
        for line in preview_lines:
            print(line)
    
    def _offer_open_folder(self, filepath):
        """Offer to open the folder containing the saved file"""
        if input("\nWould you like to open the folder containing the file? (y/n): ").lower() == 'y':
            try:
                folder_path = os.path.dirname(filepath)
                if os.name == 'nt':  # Windows
                    os.startfile(folder_path)
                else:  # macOS or Linux
                    import subprocess
                    if os.name == 'posix':
                        if 'darwin' in os.sys.platform:  # macOS
                            subprocess.call(['open', folder_path])
                        else:  # Linux
                            subprocess.call(['xdg-open', folder_path])
                print(f"Opened folder: {folder_path}")
            except Exception as e:
                print(f"Could not open folder: {e}")


class TimeTableGenerator:
    def __init__(self):
        self.llm = None
        self.llm_provider = None
        self.conversation_history = []
        self.user_data = {
            "institute_name": "",
            "date_range": {},
            "faculty": [],
            "subjects": [],
            "time_slots": [],
            "batch_names": [],
            "constraints": {}
        }
        self.csv_saver = CsvSaver()  # Initialize the CSV Saver

    def setup_llm(self):
        """Let user select which LLM provider to use"""
        print("\n===== LLM SELECTION =====")
        print("1. Groq (llama-3.3-70b-versatile)")
        print("2. Ollama - gemma3:4b (local) - 4 threads")

        while True:
            choice = input("\nSelect LLM provider (1 or 2): ")
            if choice == "1":
                self.llm_provider = "groq"
                # Check if GROQ_API_KEY is in environment
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    api_key = input("Enter your Groq API key: ")
                    os.environ["GROQ_API_KEY"] = api_key

                self.llm = Groq(api_key=api_key)
                print("Groq LLM (llama-3.3-70b-versatile) selected successfully!")
                break
            elif choice == "2":
                self.llm_provider = "ollama"
                # Check if Ollama is running
                print("Checking if Ollama is available...")
                try:
                    # Set Ollama to use 4 threads with updated OllamaLLM class
                    self.llm = OllamaLLM(
                        model="llama3.2:1b",
                        num_thread=4,  # Specify 4 threads for CPU utilization
                        num_gpu=0,     # Use CPU only (0 GPU)
                    )
                    print("Ollama with gemma3:4b model selected successfully!")
                    print("CPU threads set to: 4")
                    break
                except Exception as e:
                    print(f"Error connecting to Ollama: {e}")
                    print("Please make sure Ollama is installed and running, and gemma3:4b model is available.")
                    retry = input("Retry? (y/n): ")
                    if retry.lower() != 'y':
                        print("Please install Ollama or choose Groq instead.")
            else:
                print("Invalid choice. Please select 1 or 2.")

    def add_to_conversation(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def get_llm_response(self, prompt):
        """Get response from the selected LLM"""
        self.add_to_conversation("user", prompt)

        if self.llm_provider == "groq":
            # Use Groq API with specified model
            chat_completion = self.llm.chat.completions.create(
                messages=self.conversation_history,
                model="compound-beta",  # Use the specified model
                temperature=0.1,
                max_tokens=4096
            )
            response = chat_completion.choices[0].message.content
        else:
            # Use Ollama with gemma3:4b 
            # (with 4 threads as configured in setup_llm)
            formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
            
            # Additional parameters could be added here if needed
            response = self.llm(formatted_prompt)

        self.add_to_conversation("assistant", response)
        return response

    def extract_json_from_response(self, response, default=None):
        """Extract JSON from LLM response if present"""
        try:
            # Find JSON content between triple backticks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].strip()
                try:
                    return json.loads(json_str)
                except:
                    pass

            # Try to find anything that looks like JSON
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except:
            pass

        return default
        
    def create_reference_document(self, data, filename="institute_reference.txt"):
        """Create a reference document with institute information, faculty, and subjects"""
        try:
            content = f"""INSTITUTE INFORMATION REFERENCE
    
    Institute Name: {data['institute_name']}
    
    BATCHES:
    {', '.join(data['batch_names'])}
    
    TIME SLOTS:
    {', '.join(data['time_slots'])}
    
    SUBJECTS:
    {', '.join(data['subjects'])}
    
    FACULTY AND THEIR EXPERTISE:
    """
            
            # Add faculty information
            for faculty in data['faculty']:
                content += f"{faculty['name']} - {', '.join(faculty['subjects'])}\n"
            
            # Write to file
            with open(filename, 'w') as f:
                f.write(content)
                
            print(f"\nReference document created: {filename}")
            return filename
        except Exception as e:
            print(f"Error creating reference document: {e}")
            return None
            
    def initialize_with_reference(self, reference_path):
        """Initialize the conversation with reference information"""
        try:
            with open(reference_path, 'r') as f:
                reference_content = f.read()
                
            # Create a prompt that incorporates the reference information
            init_prompt = f"""
            You are assisting in creating a timetable. Here is some reference information about the institute, faculty, and subjects:
            
            {reference_content}
            
            Please use this information when you gather details from the user to create the timetable. 
            When asking questions, acknowledge that you already know the institute name is Thinkplus and you have the faculty information.
            Start by asking about the specific requirements for the new timetable.
            """
            
            # Add this to conversation history
            self.add_to_conversation("system", init_prompt)
            
            # Update the user_data with the reference information
            self.user_data["institute_name"] = "Thinkplus"
            
            return True
        except Exception as e:
            print(f"Error initializing with reference: {e}")
            return False
            
    def validate_date_range(self, csv_content, expected_start_date, expected_end_date):
        """Validate that the CSV covers the entire date range from start to end date"""
        try:
            # Split the CSV content into lines
            lines = csv_content.strip().split('\n')
            
            # Skip header rows (assumed to be 3 rows based on templates)
            data_lines = lines[3:]
            
            # Pattern for date extraction (e.g., "24-Mar" or "15-May")
            date_pattern = re.compile(r'^(\d{1,2}-\w{3})')
            
            # Extract dates from the first column of each row where a date appears
            timetable_dates = []
            for line in data_lines:
                if not line.strip():
                    continue
    
                parts = line.split(',')
                if not parts:
                    continue
    
                first_cell = parts[0].strip()
                match = date_pattern.search(first_cell)
                if match:
                    timetable_dates.append(match.group(1))
            
            if not timetable_dates:
                print("No dates found in the timetable.")
                return False
    
            print(f"Found dates in timetable: {timetable_dates[0]} to {timetable_dates[-1]}")
            
            # Try different date formats since input dates could be in various formats
            # Expected format in CSV is typically DD-MMM (e.g., 24-Mar)
            possible_formats = [
                "%d-%b", "%d-%B", "%d %b", "%d %B",  # 24-Mar, 24-March, 24 Mar, 24 March
                "%b-%d", "%B-%d", "%b %d", "%B %d",  # Mar-24, March-24, Mar 24, March 24
                "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",  # Various numeric formats
                "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
                "%d.%m.%Y", "%m.%d.%Y", "%Y.%m.%d"
            ]
            
            # Convert input dates to the same format as timetable dates
            formatted_start = None
            formatted_end = None
            
            # First try to parse the timetable dates to determine their format
            timetable_format = None
            for fmt in ["%d-%b", "%d-%B"]:
                try:
                    datetime.strptime(timetable_dates[0], fmt)
                    timetable_format = fmt
                    break
                except ValueError:
                    continue
            
            if not timetable_format:
                print("Could not determine timetable date format.")
                return False
            
            # Then try to convert the input dates to the timetable format
            for fmt in possible_formats:
                try:
                    start_dt = datetime.strptime(expected_start_date, fmt)
                    formatted_start = start_dt.strftime(timetable_format)
                    
                    end_dt = datetime.strptime(expected_end_date, fmt)
                    formatted_end = end_dt.strftime(timetable_format)
                    
                    break
                except ValueError:
                    continue
            
            if not formatted_start or not formatted_end:
                # If standard formats fail, try a more flexible approach with dateutil
                try:
                    from dateutil import parser
                    
                    start_dt = parser.parse(expected_start_date)
                    formatted_start = start_dt.strftime(timetable_format)
                    
                    end_dt = parser.parse(expected_end_date)
                    formatted_end = end_dt.strftime(timetable_format)
                except:
                    print(f"Could not parse input dates: {expected_start_date}, {expected_end_date}")
                    return False
            
            print(f"Expected date range: {formatted_start} to {formatted_end}")
            
            # Check if the first and last dates match the expected range
            if timetable_dates[0] != formatted_start:
                print(f"Timetable does not start on the expected date. Expected: {formatted_start}, Found: {timetable_dates[0]}")
                return False
    
            if timetable_dates[-1] != formatted_end:
                print(f"Timetable does not end on the expected date. Expected: {formatted_end}, Found: {timetable_dates[-1]}")
                return False
    
            return True
            
        except Exception as e:
            print(f"Date range validation error: {str(e)}")
            return False

    def extract_csv_from_response(self, response):
        """Extract CSV content from LLM response"""
        try:
            # Try to extract CSV from code blocks
            if "```csv" in response:
                csv_content = response.split("```csv")[1].split("```")[0].strip()
                return csv_content
            elif "```" in response and "," in response:
                csv_content = response.split("```")[1].strip()
                return csv_content
            else:
                # Look for CSV-like content with multiple lines and commas
                lines = response.split('\n')
                csv_lines = []

                # More advanced filtering for CSV-like content
                is_csv_section = False
                for line in lines:
                    # Skip empty lines at the beginning
                    if not line.strip() and not is_csv_section:
                        continue

                    # Skip lines that look like explanatory text
                    if line.startswith(('I ', 'Here ', 'This ', 'The ', 'Above ', 'Below ')) and not is_csv_section:
                        continue

                    # If we find a line with multiple commas, we're likely in the CSV section
                    if line.count(',') >= 3:
                        is_csv_section = True
                        csv_lines.append(line)
                    # Continue adding lines as long as they seem to be part of the CSV
                    elif is_csv_section and line.strip() and ',' in line:
                        csv_lines.append(line)
                    # If we've already started collecting CSV and hit a non-CSV line, we're done
                    elif is_csv_section and (not line.strip() or ',' not in line):
                        # Allow for a few blank lines within the CSV
                        if not line.strip() and len(csv_lines) > 0:
                            continue
                        else:
                            break

                if csv_lines:
                    return '\n'.join(csv_lines)
        except Exception as e:
            print(f"Error extracting CSV content: {e}")

        return None

    def validate_csv(self, csv_content):
        """Validate if the content is a valid CSV"""
        try:
            # Convert string to file-like object for csv.reader
            from io import StringIO
            csv_file = StringIO(csv_content)
            reader = csv.reader(csv_file)

            # Read all rows
            rows = list(reader)

            # Basic validation: at least 3 rows and consistent number of columns
            if len(rows) < 3:
                return False

            # Check if all rows have the same number of columns
            col_count = len(rows[0])
            for row in rows[1:]:
                if len(row) != col_count:
                    return False

            return True
        except:
            return False
            
    def load_timetable_template(self, template_path):
        """Load an existing timetable template from CSV file"""
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Parse CSV to extract basic structure and metadata
            df = pd.read_csv(template_path, header=None)
            
            # Try to identify time slots, batches, and other metadata from the template
            template_info = {
                "structure": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "header_rows": 3  # Typically first 3 rows are headers
                },
                "content": template_content
            }
            
            print(f"\nSuccessfully loaded template from {template_path}")
            print(f"Template has {template_info['structure']['rows']} rows and {template_info['structure']['columns']} columns")
            
            return template_info
        except Exception as e:
            print(f"Error loading template: {e}")
            return None
            
    def extract_data_from_template(self, template_path):
        """Extract faculty, subjects, and other relevant information from the template"""
        try:
            # Read CSV file into a DataFrame
            df = pd.read_csv(template_path, header=None)
            
            # Extract institute information
            # Usually the institute name might be in the file name or headers
            institute_name = "Thinkplus"  # Default based on requirements
            
            # Extract batch names - typically in row 1
            batch_names = []
            for col in range(3, len(df.iloc[1])):
                if isinstance(df.iloc[1, col], str) and df.iloc[1, col].strip() and 'ST' in df.iloc[1, col]:
                    if df.iloc[1, col] not in batch_names:
                        batch_names.append(df.iloc[1, col])
            
            # Extract time slots - typically in row 2
            time_slots = []
            for col in range(3, len(df.iloc[2])):
                if isinstance(df.iloc[2, col], str) and df.iloc[2, col].strip() and 'AM' in df.iloc[2, col] or 'PM' in df.iloc[2, col]:
                    time_slots.append(df.iloc[2, col])
            
            # Extract faculty names and their subjects
            faculty_data = {}
            subjects = set()
            
            # Process data rows (start from row 3, after headers)
            for row in range(3, len(df)):
                # Every third row contains faculty names
                if row % 3 == 2:  # Faculty row
                    for col in range(2, len(df.iloc[row])):
                        faculty_name = str(df.iloc[row, col]).strip()
                        if faculty_name and faculty_name != 'nan':
                            # Look up the subject from the first row of this block
                            subject = str(df.iloc[row-2, col]).strip()
                            if subject and subject != 'nan':
                                subjects.add(subject)
                                if faculty_name in faculty_data:
                                    if subject not in faculty_data[faculty_name]:
                                        faculty_data[faculty_name].append(subject)
                                else:
                                    faculty_data[faculty_name] = [subject]
            
            # Format faculty information
            faculty_list = []
            for faculty, faculty_subjects in faculty_data.items():
                faculty_list.append({
                    "name": faculty,
                    "subjects": faculty_subjects
                })
            
            # Create data structure
            extracted_data = {
                "institute_name": institute_name,
                "batch_names": batch_names,
                "faculty": faculty_list,
                "subjects": list(subjects),
                "time_slots": time_slots
            }
            
            return extracted_data
        except Exception as e:
            print(f"Error extracting data from template: {e}")
            return {
                "institute_name": "Thinkplus",
                "batch_names": ["ST-1", "ST-2"],
                "faculty": [
                    {"name": "MK", "subjects": ["QA", "LR"]},
                    {"name": "JMS", "subjects": ["QA"]},
                    {"name": "Anish", "subjects": ["VA"]},
                    {"name": "Jaya Chandra", "subjects": ["LR"]},
                    {"name": "Kiran", "subjects": ["VA", "QA"]},
                    {"name": "Rakesh", "subjects": ["QA", "VA"]},
                    {"name": "RAM", "subjects": ["QA"]},
                    {"name": "Paandu", "subjects": ["VA"]},
                    {"name": "Siddhik", "subjects": ["QA"]}
                ],
                "subjects": ["QA", "VA", "LR"],
                "time_slots": ["05:30 A.M - 06:30 A.M", "08:30 AM - 10:00 AM", "10:10 AM - 11:40 AM", "11:50 AM - 01:20 PM", "02:20 PM - 03:50 PM", "04:00 PM - 05:15 PM", "05:30 PM - 07:00 PM"]
            }

    def load_dates(self, start_date, end_date):
        """Load and update date information without validation"""
        # Update user data with the date range
        self.user_data["date_range"] = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        # Add dates to conversation history to inform the LLM
        date_info_prompt = f"""
        The user has provided the following date range for the timetable:
        Start date: {start_date}
        End date: {end_date}
        
        When generating the timetable, make sure to create entries for each date in this range.
        """
        
        self.add_to_conversation("system", date_info_prompt)
        
        return True
        
    
    def conversation_loop(self):
        """Main conversation loop to collect data from user"""
        # Check if we already have initial data
        has_initial_data = len(self.conversation_history) > 0 and self.user_data.get("institute_name")
        
        if not has_initial_data:
            # Initial greeting and explanation
            initial_prompt = """
            You are a helpful assistant that will gather information to create a timetable for an educational institute. 
            Ask the user about:
            1. Institute name
            2. Start and end dates for the timetable (accept dates in ANY format the user provides)
            3. Batch names (e.g., ST-1, ST-2)
            4. Faculty information (names and their subject expertise)
            5. Subjects to be covered
            6. Available time slots
            7. Any specific constraints or faculty availability
    
            Ask ONE question at a time in a conversational manner. After the user responds to each question, 
            acknowledge their answer and then ask the next relevant question. Start by introducing yourself and asking about the institute name.
            """
            
            print("\n===== STARTING CONVERSATION WITH LLM =====")
            print("(The LLM will ask you a series of questions to gather information for the timetable.)")
            print("-----------------------------------------------")
            
            response = self.get_llm_response(initial_prompt)
        else:
            # We already have some information from the reference document
            print("\n===== STARTING CONVERSATION WITH LLM =====")
            print("(The LLM already has basic institute information and will ask for specific details.)")
            print("-----------------------------------------------")
            
            # Just get the first response from existing conversation history
            response = self.conversation_history[-1]["content"]
        
        print(f"\nLLM: {response}")

        # Continue conversation until user wants to generate timetable
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nExiting conversation.")
                return False

            # Check if user wants to generate timetable
            if "generate" in user_input.lower() and "timetable" in user_input.lower():
                generate_prompt = """
                Based on all the information collected so far, please summarize what you know about:
                1. Institute name
                2. Date range for the timetable
                3. Batch names
                4. Faculty members and their expertise
                5. Subjects and topics
                6. Time slots
                7. Any constraints or faculty availability

                Please format this information as a JSON object with these categories and ask the user if this information is correct
                before generating the timetable.
                """

                # Get summary from LLM
                response = self.get_llm_response(generate_prompt)
                print(f"\nLLM: {response}")

                # Extract data from JSON if present
                extracted_data = self.extract_json_from_response(response)
                if extracted_data:
                    self.user_data.update(extracted_data)

                # Ask user to confirm
                confirmation = input("\nIs this information correct? (yes/no): ")
                if confirmation.lower() in ['yes', 'y']:
                    return True  # Ready to generate timetable
                else:
                    correction = input("\nWhat would you like to correct? ")
                    self.get_llm_response(
                        f"The user wants to correct: {correction}. Please ask for the correct information.")
                    print(f"\nLLM: {self.conversation_history[-1]['content']}")
                    return False  # Not ready to generate timetable, needs correction

    def generate_timetable(self, template_path=None):
        """Generate timetable using the LLM based on collected information"""
        print("\n===== GENERATING TIMETABLE =====")
        print("Using collected information to generate timetable...")
        
        # Use the default timetables folder
        if not os.path.exists(self.csv_saver.default_folder):
            os.makedirs(self.csv_saver.default_folder)
            print(f"Created timetables folder: {os.path.abspath(self.csv_saver.default_folder)}")
        
        template_info = None
        if template_path:
            template_info = self.load_timetable_template(template_path)
            print("Using existing timetable as template")
        
        # Determine start and end dates for the new timetable
        start_date = self.user_data.get("date_range", {}).get("start_date")
        end_date = self.user_data.get("date_range", {}).get("end_date")
        
        # Ensure we have dates
        if not start_date or not end_date:
            print("\nWarning: Date range is missing or incomplete.")
            if not start_date:
                start_date = input("Please enter start date (any format): ")
            if not end_date:
                end_date = input("Please enter end date (any format): ")
            
            self.user_data["date_range"] = {
                "start_date": start_date,
                "end_date": end_date
            }
        
        # Let the user know we've processed the dates
        print(f"\nGenerating timetable from {start_date} to {end_date}")
        
        # If template exists and dates are provided, create a targeted prompt
        if template_info and start_date and end_date:
            generation_prompt = f"""
            Please create a new timetable based on the provided template CSV, for the date range from {start_date} to {end_date}.
            You MUST first interpret these dates correctly, regardless of their format.
            
            Follow EXACTLY the same format as the template, including:
            - The same number of header rows and columns
            - The same batch structure and time slots
            - The same overall layout and organization
            
            Here is the template CSV content to use as reference:
            
            ```csv
            {template_info['content']}
            ```
            
            CRITICAL REQUIREMENTS:
            1. The timetable MUST cover the ENTIRE period from {start_date} to {end_date} WITHOUT EXCEPTION
            2. You need to interpret these dates correctly and convert them to DD-MMM format (e.g., 24-Mar)
            3. Each date in the timetable MUST be consecutive days within this date range 
            4. Day of the week must be accurate for each date
            5. Maintain EXACTLY the same structure as the template
            6. Keep the same time slots and batch structure
            7. Use the faculty expertise information to assign appropriate teachers
            8. IMPORTANT: You MUST include ALL dates from {start_date} through {end_date} inclusive
            9. IMPORTANT: The timetable is INCOMPLETE if it doesn't reach {end_date}
            
            Institute: {self.user_data.get("institute_name", "Your Institute")}
            Faculty: {json.dumps(self.user_data.get("faculty", []))}
            Subjects: {json.dumps(self.user_data.get("subjects", []))}
            Constraints: {json.dumps(self.user_data.get("constraints", {}))}
            
            Return ONLY the CSV content without any additional text or explanation. Format everything as valid CSV that can be saved directly to a file.
            """
        else:
            # Use the original prompt if no template or dates are provided
            generation_prompt = """
            Based on all the information we've collected, please generate a detailed timetable in CSV format. 
            Here's a summary of what we know:
    
            Institute: {institute_name}
            Date Range: {date_range}
            Batches: {batches}
            Faculty: {faculty}
            Subjects: {subjects}
            Time Slots: {time_slots}
            Constraints: {constraints}
    
            CRITICAL REQUIREMENTS:
            1. The timetable MUST start on {start_date} and end on {end_date}
            2. The timetable MUST cover the ENTIRE date range from {start_date} to {end_date} WITHOUT EXCEPTION
            3. IMPORTANT: You MUST include ALL dates from {start_date} through {end_date} inclusive
            4. IMPORTANT: The timetable is INCOMPLETE if it doesn't reach {end_date}
            5. Each date in the timetable MUST be consecutive days within this range
            6. Dates must be in DD-MMM format (e.g., 24-Mar) in the first column
            7. Day of the week must be accurate for each date
            
            I need you to create a timetable with this exact format:
            ```
            Day,Week,CONCEPT BUILDER,Short Term (ST -1),,,,,,
            ,,Test,Test,,,,,,
            ,,05:30 A.M - 06:30 A.M,08:30 AM - 10:00 AM,10:10 AM - 11:40 AM,11:50 AM - 01:20 PM,02:20 PM - 03:50 PM,04:00 PM - 05:15 PM,05:30 PM - 07:00 PM
            24-Mar,Monday,QA,QA,VA,LR,QA,LR,QA/VA
            ,,Profit and Loss -3,Si & CI - 1,"Nouns,Adverbs",Coding and Decoding,SI & CI - 2,Calendars -1,Weekend Exam -1 Paper doubts
            ,,MK,JMS,Anish,Jaya Chandra,JMS,MK,Kiran / Rakesh
            ```
    
            Additional rules:
            1. For each time slot, specify the subject (QA, VA, LR, etc.)
            2. Under each subject, specify the topic being covered
            3. Under each topic, specify the faculty member assigned
            4. Ensure faculty members are only assigned to subjects matching their expertise
            5. Try to balance the subjects across the week
    
            Return ONLY the CSV content without any additional text or explanation. Format everything as valid CSV that can be saved directly to a file.
            """.format(
                institute_name=self.user_data.get("institute_name", "Your Institute"),
                date_range=json.dumps(self.user_data.get("date_range", {})),
                start_date=start_date,
                end_date=end_date,
                batches=json.dumps(self.user_data.get("batch_names", [])),
                faculty=json.dumps(self.user_data.get("faculty", [])),
                subjects=json.dumps(self.user_data.get("subjects", [])),
                time_slots=json.dumps(self.user_data.get("time_slots", [])),
                constraints=json.dumps(self.user_data.get("constraints", {}))
            )
    
        # Get timetable from LLM (may need multiple attempts for complex schedules)
        max_attempts = 5  # Increased from 3 to 5 to give more chances to get it right
        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}...")
        
            response = self.get_llm_response(generation_prompt)
            csv_content = self.extract_csv_from_response(response)
        
            # First check if the CSV format is valid
            if csv_content and self.validate_csv(csv_content):
                # Then validate that it covers the entire date range
                date_range_valid = self.validate_date_range(csv_content, start_date, end_date)
                
                if date_range_valid:
                    # Use the CsvSaver to save the generated timetable
                    institute_name = self.user_data.get("institute_name", "Institute")
                    batch_info = self.user_data.get("batch_names", [])
                    
                    # Save the CSV file using the CsvSaver
                    success, filepath, error = self.csv_saver.save_csv(
                        csv_content=csv_content,
                        institute_name=institute_name,
                        batch_info=batch_info,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Return success or failure
                    if success:
                        print("\nTimetable generation and saving complete!")
                        return True
                    else:
                        print(f"\nError saving timetable: {error}")
                        return False
                else:
                    print("\nThe generated timetable does not cover the entire requested date range. Retrying...")
                    
                    # Add more specific instructions about date range
                    generation_prompt += f"""
        
                    CRITICAL: The previous attempt did not cover the ENTIRE date range correctly.
                    The timetable MUST:
                    1. Start EXACTLY on {start_date}
                    2. End EXACTLY on {end_date}
                    3. Include ALL dates from {start_date} to {end_date} inclusive
                    4. Have the correct day of the week for each date
                    
                    This is the MOST IMPORTANT requirement. The timetable is incomplete if it doesn't 
                    include entries for every day in the requested date range.
                    
                    Return ONLY the CSV content with the complete date range.
                    """
            else:
                print("\nThe generated timetable was not in valid CSV format. Retrying...")
        
                # Provide more specific instructions for the next attempt
                generation_prompt += f"""
        
                The previous attempt did not produce a valid CSV. Please ensure:
                1. No explanatory text before or after the CSV content
                2. Consistent number of columns in each row
                3. Proper CSV formatting with commas as separators
                4. At least 3 rows (headers + data)
                5. The first date in the timetable MUST be {start_date}
                6. The last date in the timetable MUST be {end_date}
                7. EVERY date from {start_date} to {end_date} must be included

                Return ONLY the CSV content.
                """

        print("\nFailed to generate a valid timetable after multiple attempts.")
        
        # Add a suggestion for manual verification
        print("\nPlease verify that the date range is specified in a recognizable format:")
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        print("\nTip: Consider formatting dates as 'DD-MM-YYYY' or 'YYYY-MM-DD' for best results.")
        
        # Try to install dateutil if it's not available
        try:
            import pip
            pip.main(['install', 'python-dateutil'])
            print("Installed python-dateutil package for better date parsing.")
        except:
            print("Note: For better date parsing, consider installing the python-dateutil package.")
        
        return False


def main():
    print("===== TIMETABLE GENERATOR WITH LLM =====")
    print("This program will use an LLM to help you create a timetable for your institute.")

    generator = TimeTableGenerator()
    generator.setup_llm()

    # Default template path
    default_template = "IPMAT ST Time Table (2025-2026) - 24th Mar- 29th Mar.csv"
    
    # Extract and create reference document from template
    if os.path.exists(default_template):
        print("\nExtracting institute information from template...")
        reference_data = generator.extract_data_from_template(default_template)
        reference_path = generator.create_reference_document(reference_data)
        if reference_path:
            print("Institute reference document created successfully.")
            # Initialize the conversation with reference information
            generator.initialize_with_reference(reference_path)
    
    # Ask if user wants to use an existing template
    use_template = input("\nDo you want to use an existing timetable as a template? (y/n): ").lower()
    template_path = None
    
    if use_template == 'y':
        template_path = default_template
        
        # Verify the template file exists
        if not os.path.exists(template_path):
            print(f"Warning: Template file '{template_path}' not found. Will proceed without template.")
            template_path = None
        else:
            print(f"Using template file: '{template_path}'")
            
            # If using template, ask for date range directly
            print("\nPlease provide the date range for the new timetable:")
            
            # Simple date input - accept any format
            start_date = input("Start date (any format, e.g., '15 May, 2025' or '15-05-2025'): ")
            end_date = input("End date (any format, e.g., '15 June, 2025' or '15-06-2025'): ")
            
            # Update user_data with date range using dedicated method
            generator.load_dates(start_date, end_date)
            
            # Combine with reference data if exists
            if 'reference_data' in locals() and reference_data:
                generator.user_data.update({
                    "institute_name": reference_data["institute_name"],
                    "batch_names": reference_data["batch_names"],
                    "faculty": reference_data["faculty"],
                    "subjects": reference_data["subjects"],
                    "time_slots": reference_data["time_slots"]
                })
            
            # Ask for any additional constraints or changes
            additional_info = input("\nDo you want to provide additional information about faculty, subjects, or constraints? (y/n): ").lower()
            if additional_info == 'y':
                # Continue with conversation loop to gather more info
                ready_to_generate = generator.conversation_loop()
            else:
                ready_to_generate = True
    else:
        # Regular conversation flow for creating from scratch
        ready_to_generate = generator.conversation_loop()

    if ready_to_generate:
        success = generator.generate_timetable(template_path)
        if success:
            print("\nTimetable generation complete! Your timetable has been saved successfully.")
        else:
            print("\nTimetable generation encountered issues. Please check the error messages above.")

    print("\nThank you for using the Timetable Generator!")


if __name__ == "__main__":
    main()