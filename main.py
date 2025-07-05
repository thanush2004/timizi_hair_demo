# app.py
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import inspect
import mimetypes
import re
from PIL import Image # For potentially processing image files
import io # For working with image bytes

# Gradio client imports
from gradio_client import Client, handle_file

# Google Generative AI imports
import google.generativeai as genai


load_dotenv() # Load environment variables from .env file

app = Flask(__name__)

# --- Configuration from .env ---
# Get the directory of the current script (app.py)
current_script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Construct the path to the .env file, which is one level up (in D:\Projects\timizi\backend\)
dotenv_path = os.path.join(current_script_dir, '../.env')
load_dotenv(dotenv_path=dotenv_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Error: Hugging Face API Token (HF_TOKEN) not found in environment variables.")
    exit(1)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: Google AI Studio API Key (GOOGLE_API_KEY) not found in environment variables.")
    exit(1)

# --- Initialize Google Generative AI ---
gemini_model = None
def configure_gemini():
    global gemini_model
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # UPDATED MODEL NAME: Using gemini-1.5-flash as gemini-pro-vision is deprecated
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Successfully configured Google Generative AI and loaded gemini-1.5-flash model!")
    except Exception as e:
        print(f"Failed to configure Google Generative AI: {e}")
        gemini_model = None

# Configure Gemini on startup
configure_gemini()

# --- Initialize Gradio Client ---
gradio_client = None

def initialize_gradio_client():
    global gradio_client
    try:
        print("Attempting to connect to Gradio client for Julienajd/BaldnessDetector...")
        gradio_client = Client("Julienajd/BaldnessDetector", hf_token=HF_TOKEN)
        print("Successfully connected to Gradio client!")
    except Exception as e:
        print(f"Failed to connect to Gradio client: {e}")
        gradio_client = None

# Initialize Gradio client on startup
initialize_gradio_client()

# --- Middleware to check client initialization ---
@app.before_request
def check_clients():
    global gradio_client, gemini_model
    if not gradio_client:
        initialize_gradio_client()
        if not gradio_client:
            return jsonify({"error": "Gradio client not initialized. Please try again later."}), 503
    if not gemini_model:
        configure_gemini() # Try to re-configure Gemini if it failed
        if not gemini_model:
            return jsonify({"error": "Google AI Studio model not initialized. Please check API key."}), 503

@app.route('/predict-hair', methods=['POST'])
def predict_hair():
    data = request.json
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({"error": "imageUrl is required"}), 400

    print(f"Received request for hair prediction with image URL: {image_url}")

    norwood_scale = "N/A"
    hair_segmentation_image_data = None
    mime_type = None

    try:
        # Step 1: Call Gradio BaldnessDetector
        print("Calling Gradio BaldnessDetector...")
        gradio_result = gradio_client.predict(
            filepath=handle_file(image_url), # handle_file can take URL
            api_name="/predict"
        )
        print(f"Gradio Client prediction raw result: {gradio_result}")

        # Gradio result is a tuple/list: (Norwood_Scale, Image_Path)
        norwood_scale = str(gradio_result[0])
        segmentation_image_path = gradio_result[1] # This is a local temp file path

        print(f"Norwood Scale: {norwood_scale}, Segmentation Image Path: {segmentation_image_path}")

        # Step 2: Read the segmentation image file into bytes for Gemini
        if os.path.exists(segmentation_image_path):
            with open(segmentation_image_path, 'rb') as img_file:
                hair_segmentation_image_data = img_file.read()

            mime_type, _ = mimetypes.guess_type(segmentation_image_path)
            if not mime_type: # Fallback for unknown mime types
                mime_type = 'image/jpeg' # Most common web image types work
            print(f"Segmentation image read successfully. MIME type: {mime_type}")
        else:
            print(f"Warning: Segmentation image file not found at {segmentation_image_path}. Gemini will not receive image.")
            # hair_segmentation_image_data remains None

    except Exception as e:
        print(f"Error during Gradio prediction: {e}")
        import traceback
        traceback.print_exc()
        # Continue to Gemini step, but note the Gradio failure
        norwood_scale = "N/A (Gradio failed)"
        hair_segmentation_image_data = None
        mime_type = None


    # Step 3: Call Google AI Studio (Gemini) with combined information and specific prompt
    gemini_symptoms = "N/A"
    gemini_main_issues = []
    gemini_overall_health_percentage = "N/A"
    gemini_causes = "N/A"
    gemini_treatments = "N/A"
    gemini_raw_insight = "Failed to generate detailed insight."

    try:
        if gemini_model:
            # --- UPDATED GEMINI PROMPT with 'Confidence' phrasing ---
            gemini_prompt_text = f"""
Based on a hair analysis result showing **Norwood Scale {norwood_scale}**, and considering the provided hair segmentation image (if available), generate a structured report. This report aims to provide confidence in the hair analysis.

**SYMPTOMS:**
Describe the typical observable symptoms for Norwood Scale {norwood_scale}. Max 2 lines.

**MAIN ISSUES (Confidence in Specific Concerns):**
List up to 4 primary hair loss issues commonly associated with this Norwood Scale. For each, provide *only* the issue name and an estimated percentage, formatted strictly as "Issue Name (Percentage%)". These are AI estimations. No conversational text or extra explanations for each issue.

**OVERALL HAIR HEALTH PERCENTAGE (Overall Confidence):**
Based on Norwood Scale {norwood_scale}, provide an estimated overall hair health percentage, e.g., "95%". This is an AI estimation of the overall condition.

**CAUSES:**
Explain common underlying causes for hair loss at this Norwood Scale. Max 2 lines.

**TREATMENTS:**
List general management or treatment approaches for this Norwood Scale. Keep it concise. Max 4 lines.
            """
            # --- END UPDATED GEMINI PROMPT ---

            gemini_prompt_parts = [gemini_prompt_text]

            if hair_segmentation_image_data and mime_type:
                gemini_prompt_parts.append({
                    'mime_type': mime_type,
                    'data': hair_segmentation_image_data
                })

            print("Sending request to Google AI Studio (Gemini)...")
            gemini_response_obj = gemini_model.generate_content(gemini_prompt_parts)
            gemini_raw_insight = gemini_response_obj.text
            print(f"Gemini Raw Response:\n{gemini_raw_insight}")

            # --- IMPROVED PARSING OF GEMINI'S RESPONSE ---
            # Define sections and their corresponding keys. Ensure these match the prompt headers exactly.
            sections_mapping = {
                "SYMPTOMS:": "symptoms",
                "MAIN ISSUES (Confidence in Specific Concerns):": "main_issues",
                "OVERALL HAIR HEALTH PERCENTAGE (Overall Confidence):": "overall_health_percentage",
                "CAUSES:": "causes",
                "TREATMENTS:": "treatments"
            }
            parsed_content = {}

            temp_text = gemini_raw_insight
            # Add a unique delimiter before each known section header to help split
            # Iterate through actual headers from sections_mapping keys to replace them correctly
            for header_text_in_prompt in sections_mapping.keys():
                # Handle both bolded and non-bolded headers
                temp_text = temp_text.replace(f"**{header_text_in_prompt}**", f"---SECTION_DELIMITER---{header_text_in_prompt}")
                temp_text = temp_text.replace(f"{header_text_in_prompt}", f"---SECTION_DELIMITER---{header_text_in_prompt}")


            sections_raw = temp_text.split("---SECTION_DELIMITER---")

            for section_block in sections_raw:
                section_block = section_block.strip()
                if not section_block:
                    continue

                found_header = False
                for header_text_in_prompt, key_name in sections_mapping.items():
                    if section_block.startswith(header_text_in_prompt):
                        # Extract content after the header, and remove leading/trailing whitespace
                        content = section_block[len(header_text_in_prompt):].strip()
                        parsed_content[key_name] = content
                        found_header = True
                        break

            # Assign parsed data to variables, applying .replace('*', '').strip() for cleanliness
            gemini_symptoms = parsed_content.get('symptoms', "No symptoms insight provided.").replace('*', '').strip()
            gemini_causes = parsed_content.get('causes', "No causes insight provided.").replace('*', '').strip()
            gemini_treatments = parsed_content.get('treatments', "No treatment insight provided.").replace('*', '').strip()
            gemini_overall_health_percentage = parsed_content.get('overall_health_percentage', "N/A").replace('*', '').strip()

            # Parse Main Issues (Most Critical Part to get right)
            main_issues_raw = parsed_content.get('main_issues', "")
            gemini_main_issues = []
            # Regex to capture: Optional list marker (- or *) followed by issue name (non-greedy)
            # followed by optional space, then (percentage%), then optional extra text on the same line
            issue_line_pattern = re.compile(r"^(?:[-*]\s*)?(.+?)\s*\((\d+)%\)")

            for line in main_issues_raw.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = issue_line_pattern.match(line)
                if match:
                    issue_name = match.group(1).strip()
                    issue_percent = int(match.group(2))
                    gemini_main_issues.append({"issue": issue_name, "percentage": issue_percent})
                else:
                    # Fallback if regex fails, try to get just the text, assuming 0%
                    if "(" in line and ")" in line: # Try a last ditch effort to extract if parenthesis exist
                        try:
                            name_part = line.split('(')[0].strip()
                            percent_part = line.split('(')[1].replace('%)','').replace('%','').strip()
                            gemini_main_issues.append({"issue": name_part, "percentage": int(percent_part)})
                        except (ValueError, IndexError):
                            gemini_main_issues.append({"issue": line, "percentage": 0})
                    else:
                        gemini_main_issues.append({"issue": line, "percentage": 0})

            # Ensure there's at least one main issue if a Norwood scale was detected
            if not gemini_main_issues and norwood_scale != "N/A":
                gemini_main_issues.append({"issue": f"Hair loss consistent with Norwood Scale {norwood_scale}", "percentage": 100})

        else:
            print("Gemini model not initialized. Skipping Gemini insight generation.")

    except Exception as e:
        print(f"Error calling Google AI Studio or parsing its response: {e}")
        import traceback
        traceback.print_exc()
        gemini_raw_insight = f"Failed to generate or parse detailed insight from AI Studio: {str(e)}"
        # Set default failure messages for all parsed fields if an error occurs
        gemini_symptoms = "Failed to generate insights."
        gemini_main_issues = [{"issue": "Analysis unavailable", "percentage": 0}]
        gemini_overall_health_percentage = "N/A"
        gemini_causes = "Failed to generate insights."
        gemini_treatments = "Failed to generate insights."

    # Step 4: Structure the final response for Flutter
    # This structure is what Flutter's _callBackend receives and uses for display/storage
    structured_response = {
        "label": f"Norwood Scale: {norwood_scale}", # Main label from Gradio
        "symptoms": gemini_symptoms, # From Gemini
        "main_issues": gemini_main_issues, # From Gemini (list of dicts)
        "overall_health_percentage": gemini_overall_health_percentage, # From Gemini
        "causes": gemini_causes, # From Gemini
        "treatments": gemini_treatments, # From Gemini
        "confidences": [
            {"label": f"Norwood Scale {norwood_scale}", "score": 1.0}
        ], # This remains the Gradio model's confidence for the Norwood scale
        "gemini_raw_insight": gemini_raw_insight, # The full, raw text from Gemini for debugging
        "segmentation_image_url": segmentation_image_path # Local path on backend, not directly usable by Flutter
    }

    return jsonify(structured_response), 200

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 3002)) # Default to 3002
    app.run(debug=True, host='0.0.0.0', port=PORT)