import os
import fitz  # PyMuPDF
import pdfplumber # For table extraction
import pandas as pd
# from openai import AzureOpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import shutil
import base64 # For encoding images
import re # For sanitizing filenames
import json # For handling JSON output

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME")

generation_config = {
    "temperature": 0.1,
    "max_output_tokens": 160000,  # Gemini 1.5 Pro's default max is 8192, see note below
    "response_mime_type": "application/json", # This is the key for JSON mode
}

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config=generation_config
    # If you had a system prompt, you would add it here:
    # system_instruction="You are a helpful assistant designed to output JSON."
)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def extract_text_and_image_paths_from_pdf(pdf_path, temp_image_folder):
    doc = fitz.open(pdf_path)
    full_text = ""
    image_paths_in_temp = []
    print(f"PyMuPDF: Processing PDF '{os.path.basename(pdf_path)}' with {len(doc)} pages for text/images. Current date: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d')}") # Added current date
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text", sort=True)
        full_text += f"\n--- Page {page_num + 1} (Text Extracted by PyMuPDF) ---\n{page_text}"
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            if image_ext.lower() not in ["png", "jpg", "jpeg", "gif", "webp"]: image_ext = "png"
            temp_image_filename = f"image_p{page_num + 1}_{img_index + 1}.{image_ext}"
            temp_image_path = os.path.join(temp_image_folder, temp_image_filename)
            try:
                with open(temp_image_path, "wb") as img_file: img_file.write(image_bytes)
                image_paths_in_temp.append(temp_image_path)
            except Exception as e: print(f"   PyMuPDF: Error saving image {temp_image_filename}: {e}")
    print(f"PyMuPDF: Finished. Images extracted: {len(image_paths_in_temp)}")
    return full_text, image_paths_in_temp


def extract_tables_with_pdfplumber_to_markdown(pdf_path, chosen_strategy="text"):
    markdown_tables = []
    table_counter = 0
    print(f"pdfplumber: Extracting tables from '{os.path.basename(pdf_path)}' ({chosen_strategy} strategy)...")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                settings = {"snap_tolerance": 3, "join_tolerance": 3}
                if chosen_strategy == "text": settings.update({"vertical_strategy": "text", "horizontal_strategy": "text"})
                elif chosen_strategy == "lines": settings.update({"vertical_strategy": "lines", "horizontal_strategy": "lines"})
                else: settings.update({"vertical_strategy": "text", "horizontal_strategy": "text"}) # Default to text
                page_tables = page.extract_tables(table_settings=settings)
                if page_tables:
                    for table_data in page_tables:
                        table_counter += 1
                        if not table_data: continue
                        num_rows = len(table_data)
                        num_cols = len(table_data[0]) if num_rows > 0 and table_data[0] else 0
                        # Stricter filter for trivial tables
                        if (num_rows * num_cols) < 2 or (num_rows == 1 and num_cols <= 1) or (num_cols == 1 and num_rows <=1) : continue

                        cleaned_table = [[str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row] for row in table_data if any(str(c).strip() for c in row)]
                        if not cleaned_table or not cleaned_table[0]: continue # Ensure not empty after cleaning

                        final_num_rows = len(cleaned_table)
                        final_num_cols = len(cleaned_table[0]) if final_num_rows > 0 and cleaned_table[0] else 0
                        if (final_num_rows * final_num_cols) < 2 or (final_num_rows == 1 and final_num_cols <= 1): continue

                        try:
                            # Attempt to identify a header row more robustly
                            has_plausible_header = False
                            if final_num_rows > 1 and all(str(h).strip() for h in cleaned_table[0]):
                                if final_num_rows > 2 and cleaned_table[0] != cleaned_table[1]:
                                    has_plausible_header = True
                                elif final_num_rows == 2: # If only two rows, assume first is header if not empty
                                     has_plausible_header = True
                                elif final_num_rows == 1 and all(str(h).strip() for h in cleaned_table[0]): # Single row, treat as headerless data
                                     pass 

                            df_header = cleaned_table[0] if has_plausible_header else None
                            df_data = cleaned_table[1:] if has_plausible_header else cleaned_table
                            df = pd.DataFrame(df_data, columns=df_header)
                            if df.empty: continue # Skip empty dataframes

                            markdown_tables.append(f"\n--- Table (pdfplumber, {chosen_strategy}, ID:{table_counter}, P:{page_num+1}, {df.shape[0]}x{df.shape[1]}) ---\n{df.to_markdown(index=False)}\n")
                        except Exception as df_e:
                            print(f"     pdfplumber: Error converting table {table_counter} (P:{page_num+1}) to MD: {df_e}. Using raw conversion.")
                            # Fallback to simpler raw markdown if pandas fails
                            header_md = "| " + " | ".join(map(str, cleaned_table[0])) + " |"
                            separator_md = "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |"
                            body_md = "\n".join(["| " + " | ".join(map(str,row)) + " |" for row in cleaned_table[1:]])
                            raw_md = f"{header_md}\n{separator_md}\n{body_md}"
                            markdown_tables.append(f"\n--- Table (RAW, pdfplumber, {chosen_strategy}, ID:{table_counter}, P:{page_num+1}) ---\n{raw_md}\n")
    except Exception as e: print(f"pdfplumber: Error during table extraction: {e}")
    print(f"pdfplumber ({chosen_strategy}): Processed {len(markdown_tables)} tables.")
    return markdown_tables

def save_review_file(main_output_folder, pdf_base_filename, text_content, table_markdown_list, image_paths_in_temp_dir):
    pdf_review_folder_name = sanitize_filename(f"{pdf_base_filename}_review_output")
    pdf_review_path = os.path.join(main_output_folder, pdf_review_folder_name)
    images_subfolder = "images"
    persistent_images_path = os.path.join(pdf_review_path, images_subfolder)
    os.makedirs(persistent_images_path, exist_ok=True)
    review_md_filepath = os.path.join(pdf_review_path, f"{pdf_base_filename}_review.md")
    content = [f"# PDF Content Review: {pdf_base_filename}\n\n## Extracted Raw Text\n```text\n{text_content or 'No text extracted.'}\n```\n\n## Extracted Tables\n"] # Changed 'Empty'
    content.extend(table_markdown_list or ["No tables extracted.\n"])
    content.append("\n## Extracted Images\n")
    if image_paths_in_temp_dir:
        for i, temp_img_path in enumerate(image_paths_in_temp_dir):
            img_filename = os.path.basename(temp_img_path)
            try:
                shutil.copy2(temp_img_path, os.path.join(persistent_images_path, img_filename))
                content.append(f"### Image {i+1}: {img_filename}\n![{img_filename}]({images_subfolder}/{img_filename})\n")
            except Exception as e: content.append(f"### Image {i+1}: {img_filename} (Error copying: {e})\n")
    else: content.append("No images extracted.\n")
    try:
        with open(review_md_filepath, "w", encoding="utf-8") as f: f.write("\n".join(content))
        print(f"Review file saved to: {review_md_filepath}")
    except Exception as e: print(f"Error writing review file {review_md_filepath}: {e}")


def generate_user_stories_with_gpt4o(pdf_text_content, markdown_tables_content, image_paths_list): # Removed num_user_stories
    # if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    #     return json.dumps({"error": "Azure OpenAI config missing for User Story Generation."})
    
    print(f"\n--- Preparing content for GPT-4o Master User Story Generation ---") # Updated print
    # client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version=AZURE_OPENAI_API_VERSION)
    
    requirement_context = f"""[RAW TEXT FROM PDF DOCUMENT]
{pdf_text_content if pdf_text_content and pdf_text_content.strip() else "No raw text content was extracted or provided."}
---------------------------------
[TABLES EXTRACTED FROM PDF DOCUMENT (MARKDOWN FORMAT)]
{markdown_tables_content if markdown_tables_content and markdown_tables_content.strip() else "No tables were extracted or provided."}
"""
    system_prompt = f"""**Role**: You are an World class expert AI Product Analyst specializing in creating detailed, thorough, and covering every part of context very high-quality user stories. Your primary goal is to accurately translate Product Requirement Document (PRD) content into detailed, clear, step by step actionable, and well-structured even infered or small user stories.

**Task**: Analyze the provided PRD content (raw text, tables, and images). Based on this analysis, generate a comprehensive set of user stories that will Strive for comprehensive coverage of the provided context, addressing all pertinent details and minimizing omissions and my analysis should aim to thoroughly address every relevant aspect of the context and creating user stories that collectively provide the most complete and granular representation of the PRD's requirements as is reasonably possible. Each story should represent a very detailed, piece of functionality or requirements found in the PRD. Generate as many unique user stories as possile from the provided context.

*******My primary and non-negotiable task is to create a user story for every single user-facing feature and functionality present in the provided context. Every stated feature, inferred capability, small mention, and granular detail that represents a user interaction or delivers value to the user must be captured in its own story. This exhaustive focus on functionality is paramount for the project's success. Critically, this means you must differentiate between user functionality and technical implementation. Therefore, user stories must not be created for non-functional, architectural components such as Client Architecture (web/mobile), API Gateway, Microservices, Data Layer Configuration, or Monitoring and Analytics:

**Key Important Instructions for User Story Generation**:
1.  **Strict Contextual Grounding**: All user stories MUST be derived solely from the information present in the provided PRD content (text, tables, and images). Do not introduce external information or make assumptions beyond what is stated or directly implied by the PRD.
2.  **Exclusion of Non-Functional & Architectural Stories**: Do not create user stories for backend infrastructure, architectural patterns, or system configuration details. While this information is useful for understanding the project's context, it does not represent a direct user-facing functionality. Specifically, avoid creating stories for the following: Client Architecture (e.g., "Set up the iOS project"), API Gateway (e.g., "Configure routing in the gateway"), Microservices Architecture or any other system architecture, Data Layer Configuration (e.g., "Define the database schema"), Monitoring and Analytics setup (e.g., "Integrate with a logging service")
3.  **Thorough Analysis**: Carefully examine all parts of the PRD, including text, any tables, and especially the visual information in images (e.g., diagrams, mockups, flowcharts), to identify user needs and system requirements.
4.  **Identify All Distinct Stories**: Extract all unique user-facing features or requirements. The number of stories should reflect the actual distinct functionalities present.
5.  **Clarity and Understandability**:
    * **STORY_NAME**: Choose a concise, descriptive, and clear title that captures the story's main purpose.
    * **DESCRIPTION**: Write a very detailed description from a user perspective. This detailed description should be writtern in such a way that it can be easily understood by all stakeholders, including those new to the project with such detailed and clear thought out description.
    * **ACCEPTANCE_CRITERIA**: List detailed thorough specific, verifiable conditions derived from PRD details (including information from images). These criteria define when the story is considered complete and correctly implemented in very detail.
6.  **Focus on User Value**: Ensure each story is detailed and clearly communicates the benefit or value to the end-user or the system.
7.  **Logical Structure**: Internally, ensure a logical thought process to connect different pieces of information from the PRD to form cohesive and detailed user stories.

**Input Context**: The textual and tabular PRD content is provided below. Images, which are crucial for a full understanding, will be sent in the user message.

{requirement_context}

******
###Core Principles For Crafting Description & Acceptance Criteria###:
1. A core principle is to meticulously write a very detailed, exhaustive description for the user story. This narrative must be a thoughtfully crafted and comprehensive account of the user's specific context, their precise goal, and the tangible value they achieve. It must focus on the user's interaction with the system and the value they receive, deliberately excluding implementation details of the underlying architecture like microservices or data layers. It must include granular details for absolute clarity and understanding by any stakeholder.
2. Another core principle is to meticulously define a very detailed, exhaustive set of acceptance criteria for the user story. These conditions must be thoughtfully crafted and comprehensive, covering all specific, atomic, and verifiable outcomes, UI/UX behaviors, data validations, error handling, and boundary conditions. These criteria must validate the functional behavior from a user's perspective, not the underlying architectural implementation. They must include granular checks to guarantee completeness and testability for the specified feature.
******

**Required Output Format (JSON Object)**:
Your entire response MUST be a single, valid JSON object. Do not include any other text, explanations, or apologies.
The root JSON object must have one key: `"user_stories"`.
The value of `"user_stories"` MUST be an array of user story objects.
Each user story object MUST follow this schema:
`{{`
  `"SERIAL_NUMBER": <integer, starting from 1 and incrementing for each story>,`
  `"STORY_NAME": "<string: Clear and descriptive title>",`
  `"DESCRIPTION": "<string: Detailed User-centric description: Base this on all PRD sources including images.>",`
  `"ACCEPTANCE_CRITERIA": [`
    `"<string: Detailed Specific, testable criterion 1 from PRD (text, tables, images)>",`
    `"<string: Detailed Specific, testable criterion 2 from PRD (text, tables, images)>",`
    `... // Include all necessary criteria for completeness`
  `]`
`}}`
*****Key Principles & Guidelines for Achieving Maximum Story Extraction: A core principle is to ensure that even minor or highly specific functionalities identified in the PRD are captured as distinct, detailed user stories, always grounded in the provided materials. Always Focus on meticulous detail: every identifiable requirement, whether small or inferred, should be developed into a full, contextually-backed user story. Furthermore, it is crucial to include user stories based on reasonable inferences from the PRD. Please apply a diligent internal review process to maximize the inclusion of all such relevant inferred stories, ensuring they contribute to a comprehensive output and the  process should involve a thorough self-validation step to confirm that all reasonably inferable user stories are identified and incorporated. The goal is to achieve the most complete set of stories the PRD can support through careful analysis and inference. Always "Adopt a mindset of iterative refinement and comprehensive checking. This means diligently searching for and including stories that, while not explicitly stated, are strongly implied or can be logically deduced from the combined information in the PRD.*****

####
Success Criteria : For the paramount success of this project, your most critical function is to perform an exhaustive and deeply insightful analysis of every component within the provided PRD. This comprehensive examination must ensure that absolutely every potential user story is not only identified but also fully developed, carefully considering and reflecting the diverse perspectives of all mentioned entities. This means meticulously recognizing every explicit function, all logically inferred needs, each small or briefly noted or infered detail that holds any value, and all granular aspects of any behavior or interaction. Each such unique element, without exception, must then be translated into its own complete, detailed user story that clearly articulates the specific viewpoint and benefit for the relevant entity. The entire process must conclude with a final, rigorous audit against the PRD to confirm that no piece of information has been overlooked even if it is minute, thereby guaranteeing that every conceivable user story is critically captured from all pertinent angles .
####

Please proceed with generating the user stories based on these revised guidelines.
"""
    # User content parts remain largely the same, emphasizing analysis of all provided content
    # user_content_parts = [{"type": "text", "text": "As an expert AI Product Analyst, I will analyze the complete PRD context (text & tables provided in the system prompt, and the critical images provided below). My task is to generate all identifiable, contextually-bound user stories including very granular stories, and also distinct variations if they represent even very minor user needs or contextual differences evident in the PRD. Ensuring my response is covering every user story present is critically important for the final output and I will also employ a rigorous internal review process, simulating multiple validation passes, to ensure no potential story is overlooked and then giving output strictly in the specified JSON object."}]
    # if image_paths_list:
    #     user_content_parts.append({"type": "text", "text": "\n[CRITICAL IMAGES FROM PRD - ANALYZE CAREFULLY & COHERENTLY FOR REQUIREMENT IDENTIFICATION]"}) # Slightly toned down
    #     for i, img_path in enumerate(image_paths_list):
    #         base64_image = encode_image_to_base64(img_path)
    #         if base64_image:
    #             ext = os.path.splitext(img_path)[1].lower().replace('.', '') or "png"
    #             if ext == "jpg": ext = "jpeg"
    #             if ext not in ["jpeg", "png", "gif", "webp"]: ext = "png"
    #             user_content_parts.extend([
    #                 {"type": "text", "text": f"\n--- Image {i+1}: {os.path.basename(img_path)} (This image provides important details for requirements) ---"}, # Toned down
    #                 {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{base64_image}"}}
    #             ])
    #         else: user_content_parts.append({"type": "text", "text": f"\n--- Image {i+1} ({os.path.basename(img_path)}): Error encoding this image. Please base analysis on other available content. ---"})
    # else: user_content_parts.append({"type": "text", "text": "\n[No images were provided with this PRD. Base your analysis solely on the text and tables.]"})
    
    # user_content_parts.append({"type": "text", "text": "\n---------------------------------\nProvide your response as ONLY the single, valid JSON object, adhering to all schema requirements."}) # Toned down
    
    user_content_parts = []

    # Introductory message
    user_content_parts.append({
        "text": "As an expert AI Product Analyst, I will analyze the complete PRD context (text & tables provided in the system prompt, and the critical images provided below). My task is to generate all identifiable, contextually-bound user stories including very granular stories, and also distinct variations if they represent even very minor user needs or contextual differences evident in the PRD. Ensuring my response is covering every user story present is critically important for the final output and I will also employ a rigorous internal review process, simulating multiple validation passes, to ensure no potential story is overlooked and then giving output strictly in the specified JSON object."
    })

    # Add image parts if available
    if image_paths_list:
        user_content_parts.append({"text": "\n[CRITICAL IMAGES FROM PRD - ANALYZE CAREFULLY & COHERENTLY FOR REQUIREMENT IDENTIFICATION]"})
        for i, img_path in enumerate(image_paths_list):
            base64_image = encode_image_to_base64(img_path)
            if base64_image:
                ext = os.path.splitext(img_path)[1].lower().replace('.', '') or "png"
                if ext == "jpg":
                    ext = "jpeg"
                if ext not in ["jpeg", "png", "gif", "webp"]:
                    ext = "png"

                # Strip the data:image/...;base64, prefix if present
                if base64_image.startswith("data:"):
                    base64_image = base64_image.split(",", 1)[1]

                user_content_parts.append({
                    "text": f"\n--- Image {i+1}: {os.path.basename(img_path)} (This image provides important details for requirements) ---"
                })
                user_content_parts.append({
                    "inline_data": {
                        "mime_type": f"image/{ext}",
                        "data": base64_image
                    }
                })
            else:
                user_content_parts.append({
                    "text": f"\n--- Image {i+1} ({os.path.basename(img_path)}): Error encoding this image. Please base analysis on other available content. ---"
                })
    else:
        user_content_parts.append({
            "text": "\n[No images were provided with this PRD. Base your analysis solely on the text and tables.]"
        })

    # Final instruction
    user_content_parts.append({
        "text": "\n---------------------------------\nProvide your response as ONLY the single, valid JSON object, adhering to all schema requirements."
    })

    # messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content_parts}]

    # messages = {"role": "user", "parts": [{"text": user_content_parts}]}

    messages = {"role": "user", "parts": user_content_parts}
    
    result_json = "" 
    try:
        print("--- Sending content to GPT-4o for Master User Story Generation ---")
        # response = client.chat.completions.create(
        #     model=AZURE_OPENAI_DEPLOYMENT_NAME, 
        #     messages=messages, 
        #     temperature=0.1, # Very low temperature for high fidelity to context and structure
        #     max_tokens=16000, # Allow ample space for potentially many stories
        #     response_format={"type": "json_object"}
        # )
        model = genai.GenerativeModel(
                    model_name="gemini-2.5-pro",
                    generation_config=generation_config,
                    system_instruction=system_prompt
                    # If you had a system prompt, you would add it here:
                    # system_instruction="You are a helpful assistant designed to output JSON."
                )
        # print("Messages",messages)
        response =  model.generate_content(contents=messages,generation_config={"response_mime_type": "application/json"})  # to receive structured JSON back)
        # print('Received Response',response)
        # result_json = response.choices[0].message.content
        result_json = response.text.strip("```json").strip("```").strip()
        print("JSON Results",result_json)
        print("--- GPT-4o Master User Story JSON Received ---")
        json.loads(result_json) # Validate
        return result_json
    except json.JSONDecodeError as json_e:
        err_detail = f"User Story Generation: GPT-4o response was not valid JSON. Error: {json_e}. Received content: {result_json[:500]}..."
        print(err_detail)
        return json.dumps({"error": "API response for user stories was not valid JSON.", "details": str(json_e), "received_content": result_json})
    except Exception as e:
        response_text_snippet = "N/A"
        if hasattr(e, 'response') and e.response is not None:
            # Check if response.text is actually available and not None
            if hasattr(e.response, 'text') and e.response.text is not None:
                 response_text_snippet = e.response.text[:200]
            else: # If e.response.text is None or not present, try to get status code
                status_code = getattr(e.response, 'status_code', 'Unknown status')
                response_text_snippet = f"Response object present but no text content. Status: {status_code}"


        err_detail = f"Error calling Azure OpenAI GPT-4o for user stories: {type(e).__name__} - {e}. Response text snippet: {response_text_snippet}"
        print(err_detail)
        
        # More specific check for context length exceeded, potentially from the error message or a parsed body
        is_context_length_error = False
        if "context_length_exceeded" in str(e).lower():
            is_context_length_error = True
        elif hasattr(e, 'response') and e.response is not None:
             try:
                error_body = e.response.json() # This might fail if response is not JSON
                print(f"Parsed error body: {error_body}")
                if "context_length_exceeded" in str(error_body).lower() or \
                   ("error" in error_body and isinstance(error_body["error"], dict) and \
                    error_body["error"].get("code") == "context_length_exceeded"):
                    is_context_length_error = True
             except Exception as parse_exc:
                 print(f"Could not parse error response body as JSON: {parse_exc}")
        
        if is_context_length_error:
            return json.dumps({"error": "The document content (text + images) is too long for user story generation."})
            
        return json.dumps({"error": "General API error during user story generation.", "details": str(e)})


def generate_test_cases_for_user_stories_with_gpt4o(user_stories_json_string, exact_test_cases_per_story=5):
    
    # if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_VERSION]):
    #     return json.dumps({"error": "Azure OpenAI configuration missing for Test Case Generation."})

    print(f"\n--- Initializing Test Case Generation ({exact_test_cases_per_story} initial concepts per story) ---")

    try:
        user_stories_data = json.loads(user_stories_json_string)
        if "user_stories" not in user_stories_data or \
           not isinstance(user_stories_data["user_stories"], list) or \
           not user_stories_data["user_stories"]:
            return json.dumps({"error": "Invalid or empty user stories JSON for Test Case Generation."})
        
        all_input_stories = user_stories_data["user_stories"]
        total_stories = len(all_input_stories)
        print(f"--- Total user stories to process: {total_stories} ---")

    except json.JSONDecodeError as e:
        return json.dumps({"error": "Invalid user stories JSON string provided.", "details": str(e)})

    # client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY, api_version=AZURE_OPENAI_API_VERSION)

    system_prompt = f"""**Role**: You are a **Metacognitive Quality & Detail Architect**. Your fundamental purpose is to generate outputs (specifically, system test cases derived from user stories in this context) that are not merely compliant with a schema, but are paragons of clarity, exhaustive detail, profound contextual fidelity, and flawless usability, especially for individuals with no prior experience. You achieve this by internalizing, embodying, and applying the comprehensive 'Overall Philosophy for Exhaustive & High-Quality Generation,' the 'Detailed Standards for Key Parts of Generated Output,' and the 'Cross-Cutting Directive for Capturing Details' as outlined below to every facet of your generation process and output.

**Core Objective**: Your primary task is to receive input User Stories (including their full descriptions and all acceptance criteria) and, for each, generate a suite of system test cases. Your generation process for each test case and its every component (e.g., Title, Setup, Test Steps, Expected Results) MUST be rigorously governed by the principles, rules, and success targets detailed herein. Your final JSON output must be a direct testament to your adherence to these exacting standards, particularly focusing on producing the maximum number of granular, inferred, and meticulously detailed test cases.

****Critical Mandate for Foundational Test Scenarios: For every user story provided, it is mandatory to establish a baseline of precisely {exact_test_cases_per_story} distinct conceptual test scenarios for very user story. Each of these scenarios must then be meticulously developed into exceptionally detailed and comprehensive test cases, reflecting state-of-the-art quality and exhaustive thoroughness as explicitly defined within these directives.****

**I. Overall Philosophy for Exhaustive & High-Quality Generation (To Be Internalized and Applied Universally):**

    * **A. Key Principles You WILL Operate By:**
        1.  **Deep Contextual Fidelity**: All generated test cases WILL be rigorously and exclusively anchored to the source user story (including its name, full description, and all acceptance criteria). Your process is one of meticulous translation, deep inference from context, and exhaustive elaboration of the provided material, never external invention.
        2.  **Extreme Granularity & Detail**: You WILL operate on the principle of "assume nothing; explain everything explicitly." Every piece of information relevant to a test case, from preconditions to steps and expected outcomes, WILL be broken down to its most fundamental, clear, and unambiguous components. If a detail can be further specified for clarity, or an action further decomposed for precision, you MUST do so.
        3.  **Absolute Clarity for the Novice User (Noobie-Proof Standard)**: Every generated test case, and each of its constituent parts, WILL be crafted with such clarity, precision, and unambiguous language that an individual with zero prior context or experience with the application can understand its purpose and execute it flawlessly.
        4.  **Exhaustive Coverage through Diligent Discovery**: You WILL actively and persistently seek out not only explicitly stated functionalities and criteria within the user story but also all logically inferred requirements, minute operational details, and granular aspects of user interaction or system behavior relevant to it. Your goal is to leave no potential test scenario or verification point unaddressed.

    * **B. Mandatory Rules You WILL Strictly Adhere To:**
        1.  **Strict Adherence to User Story Scope**: Test case generation WILL strictly confine itself to validating the universe defined by the specific input user story. No external information or assumptions beyond what is directly stated or strongly and logically implied by that user story are permitted.
        2.  **Uncompromising Detail in Every Component**: Every field within every generated test case object WILL be populated with the highest possible level of detail, precision, and clarity. Vague, generic, or ambiguous statements are strictly forbidden.
        3.  **Proactive Identification and Articulation of Implicit Information**: Your process MUST include actively identifying, articulating, and creating test coverage for behaviors, conditions, or edge cases that are not explicitly written in the user story but can be reasonably and logically inferred from its context, stated goals, common usability expectations, or the nature of the described functionality.
        4.  **Prescribed Format Compliance with Absolute Precision**: The final output WILL strictly adhere to the specified JSON formatting requirements with no deviations.

    * **C. Success Criteria Your Output WILL Achieve:**
        1.  **100% Traceability and Exhaustive Coverage of User Story**: Every aspect of the source user story (explicit requirements, all acceptance criteria, and all reasonably inferred functionalities/conditions related to it) WILL be demonstrably and comprehensively addressed by the generated suite of test cases.
        2.  **Flawless Novice Usability**: A tester with no prior knowledge of the application WILL be able to correctly understand the purpose of each test case and execute its steps accurately without needing any external clarification.
        3.  **Zero Ambiguity**: Every statement, instruction, precondition, step, and expected result within every test case WILL possess a singular, clear meaning, allowing for no misinterpretation.
        4.  **Capture of Granular, Minute & Inferred Details**: The generated test cases WILL clearly reflect a deep forensic analysis of the user story, successfully identifying, incorporating, and validating minute operational details, granular interaction points, and logically inferred system behaviors and edge cases.

**II. Detailed Standards for Key Parts of Generated Test Cases (Application of Overall Philosophy to Specific Fields):**

    * **A. For Generating the `TITLE` Field:**
        * **Key Principles to Embody**: Your generated `TITLE` WILL reflect conciseness achieved without sacrificing descriptiveness, and it WILL provide immediate, unambiguous clarity regarding the test's core focus.
        * **Mandatory Rules to Follow**:
            1.  The `TITLE` MUST serve as a comprehensive yet succinct summary, instantly conveying the test case's specific objective, the exact scenario under examination, and any pivotal conditions or data states being validated.
            2.  If the test case is a continuation part of a split conceptual scenario, the `TITLE` MUST explicitly and clearly indicate this, referencing the primary test case or concept it continues.
            3.  The `TITLE` MUST avoid internal jargon or undefined abbreviations; its language must be universally understandable.
        * **Success Criteria for the `TITLE`**:
            1.  The generated `TITLE` is instantly and fully understandable by any team member.
            2.  It accurately and uniquely encapsulates the precise scope and specific focus of that individual test case (or part thereof).
            3.  Titles for continuation parts clearly and correctly identify their nature and relationship to the original concept.

    * **B. For Generating the `SETUP` (Preconditions) Field:**
        * **Key Principles to Embody**: Your generated `SETUP` WILL ensure absolute verifiability of each precondition, guarantee test repeatability through identical starting conditions, and define the necessary environment with zero ambiguity.
        * **Mandatory Rules to Follow**:
            1.  The `SETUP` array MUST list all extremely specific, indispensable, and individually verifiable preconditions required before test execution can validly begin.
            2.  You MUST detail the precise identities, roles, and statuses of user accounts; the exact states of specific data entities (including values of key attributes relevant to the test); specific application configurations or feature flag settings; exact navigation points or initial UI states; and any necessary files with their precise characteristics and locations.
            3.  The `SETUP` MUST NOT contain any vague or generic statements; every setup item must be an explicit, actionable, and verifiable instruction.
        * **Success Criteria for `SETUP`**:
            1.  A novice tester can use the `SETUP` instructions to prepare the test environment flawlessly, consistently, and without any need for external clarification.
            2.  All dependencies (system, data, user, environment) essential for the test's valid and unambiguous execution are explicitly stated and verifiable.
            3.  The starting state for the test is defined with such precision that there is no room for interpretation.

    * **C. For Generating the `TEST_STEPS` Field:**
        * **Key Principles to Embody**: Each step you generate WILL exhibit singular atomicity (one distinct action), unquestionable actionability (clear directive), flawless logical sequentiality, and be constructed for novice-proof execution (anticipating and addressing potential novice confusion).
        * **Mandatory Rules to Follow**:
            1.  Each string in the `TEST_STEPS` array MUST represent a single, atomic, crystal-clear, unambiguously actionable, and sequentially logical command for the tester.
            2.  You MUST explicitly identify all UI elements involved in a step using their most stable, visible, and easily recognizable identifier for a novice (e.g., full visible label, distinct placeholder text, or a clear visual/positional description if other identifiers are insufficient).
            3.  You MUST provide the exact, literal data for any inputs required by a step, including specific examples of invalid data formats or values for negative test scenarios. Avoid abstract data descriptions.
            4.  You MUST describe all navigation actions with painstaking precision, detailing the starting UI state and the exact user interactions needed to reach the subsequent state or target element.
            5.  You MUST decompose any complex user interactions or operations into a sequence of multiple, simpler, highly granular steps.
        * **Success Criteria for `TEST_STEPS`**:
            1.  Any tester, particularly a novice, can execute the steps sequentially and accurately without requiring any clarification or making any assumptions.
            2.  There is absolutely no ambiguity regarding the action to be performed, the specific UI element to interact with, or the exact data to be used in any step.
            3.  All complex interactions are successfully broken down into an easily digestible and executable sequence of simple, atomic actions.

    * **D. For Generating the `EXPECTED_RESULTS` Field:**
        * **Key Principles to Embody**: Every expected result you generate WILL be characterized by its objective observability (can be seen/read), empirical verifiability (can be checked against a defined state), direct causality (a clear consequence of test steps), and painstaking precision in its description.
        * **Mandatory Rules to Follow**:
            1.  You MUST describe all precise, objectively observable, and verifiable outcomes that are direct and logical consequences of the preceding `TEST_STEPS`.
            2.  You MUST detail with extreme specificity all exact UI changes (text content alterations, element state changes like enabled/disabled or visible/hidden, modifications to visual cues like colors or borders, appearance/disappearance of UI elements such as modals or list items, and any structural or positional shifts of elements).
            3.  You MUST instruct for the quotation of the full, exact, literal text of any and all system-generated messages (success, error, warning, informational, tooltips, confirmation dialogs), and also describe their visual appearance and precise location on the screen.
            4.  You MUST describe how changes to underlying data are reflected and become verifiably observable within the UI or through subsequent testable system behavior (e.g., updated values in displayed fields, changes in list/table contents, altered counters, modified timestamps presented to the user, specifying expected formats for dynamic data).
            5.  You MUST NOT include any generic, subjective, or vague statements; every aspect of every expected outcome must be explicitly and objectively defined.
        * **Success Criteria for `EXPECTED_RESULTS`**:
            1.  The determination of pass/fail based on the described outcomes is entirely objective, unambiguous, and can be made consistently and confidently by any tester, especially a novice.
            2.  All significant consequences (visual, textual, data-related, state-related, navigational) of the preceding test actions are documented with verifiable specifics.
            3.  The description of expected results allows for precise validation against the application's behavior, leaving absolutely no doubt as to what constitutes the successful execution of the test steps.

**III. Cross-Cutting Directive for Capturing Inferred, Minute, and Granular Details Exhaustively (To Be Applied Throughout Analysis and Generation):**

    * **A. Key Principles to Embody**:
        1.  **Proactive Discovery and Articulation**: You WILL NOT just process obvious, explicitly stated information; you WILL actively seek out, identify, and articulate the subtle, implied, and fine-grained aspects of the user story relevant to testing.
        2.  **Value-Driven Granularity in Testing**: Detail WILL be added, and granularity in test scenarios and steps WILL be pursued, wherever it enhances clarity, improves testability, or ensures a more complete validation of a distinct requirement, behavior, or potential failure point.

    * **B. Mandatory Rules to Follow**:
        1.  You MUST actively look for and derive test scenarios from requirements, behaviors, or conditions that are not explicitly stated in the user story but can be logically and reasonably inferred from its provided context, general usability principles, common sense application logic, or domain knowledge pertinent to the described features.
        2.  If a minute detail from the user story, a granular aspect of a described interaction, or a reasonably inferred need/condition can be isolated as a distinct point requiring validation, it MUST be captured. This means creating a highly focused test case (or a distinct part of a split test case if appropriate) or ensuring it's a critically specific and verifiable component within an existing test case's steps or expected results.
        3.  Any assumptions made during the inference process must be minimal, directly and logically tied to the source user story material, and specifically aimed at clarifying or completing a comprehensive test coverage of the user-centric functionality.

    * **C. Success Criteria for Capturing Details**:
        1.  The generated test cases WILL reflect a deep, comprehensive understanding of the user story that extends beyond surface-level interpretation, clearly addressing implicit needs, potential edge cases, and fine-grained system behaviors relevant to the story.
        2.  The level of granularity in the test cases (scenarios, steps, and expected results) WILL be sufficient to ensure that all distinct aspects of the user story, including subtle nuances and inferred conditions, are individually and thoroughly testable.
        3.  It WILL be evident from the test suite that a diligent and exhaustive effort was made to identify and incorporate every piece of relevant information from the user story, no matter how small, into an appropriate and effective test validation point.

**IV. Test Case Structure and Splitting (Adherence to `Key Guidelines` item 3 is Mandatory):**
    * You are to define {{exact_test_cases_per_story}} distinct conceptual test scenarios for each user story as a baseline.
    * **If a single conceptual test scenario is too complex or long** to be detailed adequately within one test case object (e.g., it would require significantly more than 8-10 hyper-detailed steps, or it involves multiple major validation facets like a positive flow plus several distinct negative paths and various UI state changes for one core feature interaction), you **MUST** represent this single concept across **multiple, sequentially linked test case objects** using the `SPLIT_INFO` field.
    * The first test case object for such a complex scenario is the "primary" part. Subsequent objects detailing further aspects or steps of THIS SAME conceptual scenario are "continuation parts."
    * This ensures exhaustive detail and means the total number of test case objects in the output array for a user story **may exceed** {{exact_test_cases_per_story}}.

*****Generating test cases for every user stories is mandatory and needs to be done at any cost.*****

**V. Required Output Format (Strict JSON - Adherence is CRITICAL)**:
Your entire response MUST be a single, valid JSON object. No other text or explanations.
The root JSON object has one key: `"all_test_suites"`.
Value of `"all_test_suites"`: an array of "test suite" objects (one per user story).
Each "test suite" object:
`{{`
  `"user_story_serial_number": <integer, from source user story>,`
  `"user_story_name": "<string, from source user story>",`
  `"test_cases": [`
    `// Array of test case objects. Total number may exceed {{exact_test_cases_per_story}} if splitting is required for exhaustive detail.`
    `{{`
      `"TEST_CASE_ID": "<string, TC_US<UserStorySerialNumber>_<SequentialNumberForAllTestObjectsForThisStory (001, 002, 003 etc.)>>",`
      `"TITLE": "<string: Highly descriptive title. If a continuation, indicate it.>",`
      `"SETUP": ["<string, MANDATORY & EXTREMELY specific precondition 1>", ...],` // Array of detailed strings
      `"TEST_STEPS": ["<string, Granular, singular, actionable step 1 with EXPLICIT UI details, navigation, and description of data to be used>", ...],` // Array of hyper-detailed strings
      `"EXPECTED_RESULTS": ["<string, Precise, observable, and THOROUGH outcome for step(s) 1 (exact UI changes, verbatim messages, data state reflections, redirects)>", ...],` // Array of hyper-detailed strings
      `"SPLIT_INFO": {{`
        `"IS_CONTINUATION_PART": <boolean>,`
        `"CONTINUATION_OF_TEST_CASE_ID": "<string>" // TEST_CASE_ID of the primary test case this entry elaborates on/continues. "None" if IS_CONTINUATION_PART is false.`
      `}}`
    `}}`
  `]`
`}}`

**VI. Instructions for Splitting (Reiteration for Emphasis - Adherence to `Key Guidelines` item 3 is Mandatory)**:
If a conceptual test is split for detail:
- The first part (e.g., "TC_US1_002") has `IS_CONTINUATION_PART: false` and `CONTINUATION_OF_TEST_CASE_ID: "None"`.
- The next part, continuing the *same concept* (now "TC_US1_003"), has `IS_CONTINUATION_PART: true` and `CONTINUATION_OF_TEST_CASE_ID: "TC_US1_002"`.
- This continues if further splitting of that *same original concept* is needed. Then you would move to the next of the {{exact_test_cases_per_story}} initial conceptual scenarios.
The goal is **maximum detail and clarity** in every field of every test case object. Your ability to produce such output, embodying all the detailed standards above, is highly valued.

**VII. Mandatory Protocol for SUCCESS CRITERIA : 
The overall success of this entire operation is measured by the comprehensive fulfillment of the following criteria. These criteria presuppose and depend upon your diligent application of all preceding directives, particularly the 'Overall Philosophy for Exhaustive & High-Quality Generation' (Section I), the 'Detailed Standards for Key Parts of Generated Test Cases' (Section II), and the 'Cross-Cutting Directive for Capturing Details' (Section III). Your generated output must fully satisfy:
-> Absolute User Story Coverage & Prescribed Scenario Foundation:
-- Every single User Story provided as input MUST be demonstrably addressed with a corresponding suite of test cases; no User Story shall be omitted under any circumstances.
-- Each User Story MUST serve as the foundation for precisely {exact_test_cases_per_story} initial distinct conceptual test scenarios. These scenarios, in turn, MUST be meticulously developed into one or more test case objects, adhering to all stipulations for extreme detail, granularity, and splitting logic (as detailed in Sections II, III, and IV).

->Unwavering Adherence to Supreme Quality & Detail Mandates:
-- The content of every generated test case (specifically its TITLE, SETUP, TEST_STEPS, and EXPECTED_RESULTS) MUST fully embody and reflect the exhaustive detail, clarity, and novice-usability standards articulated in Sections I, II, and III.
-- This includes, but is not limited to, the deep contextual fidelity, extreme granularity, capture of all inferred and minute details, and absolute clarity required for flawless novice usability and zero ambiguity.

-> Confirmation of Process Integrity & Exhaustive Completion:
-- The final output MUST stand as clear evidence that a rigorous internal validation and (if necessary) remediation process has been successfully executed.
-- This confirmation means that the dual objectives of (a) achieving universal User Story coverage with the correct number of foundational scenarios (as per Criterion 1) AND (b) upholding all mandated quality and detail standards (as per Criterion 2) have been met for every single User Story without exception.
"""
    
    chunk_size = 4
    all_generated_test_suites_concatenated = [] # To store test_suites from all chunks

    for i in range(0, total_stories, chunk_size):
        current_chunk_of_stories = all_input_stories[i:i + chunk_size]
        chunk_number = (i // chunk_size) + 1
        total_chunks = (total_stories + chunk_size - 1) // chunk_size
        
        print(f"\n--- Processing Chunk {chunk_number}/{total_chunks} ({len(current_chunk_of_stories)} stories) ---")

        stories_for_prompt_list_chunk = []
        for story in current_chunk_of_stories:
            story_text = (
                f"User Story SERIAL_NUMBER: {story.get('SERIAL_NUMBER', 'N/A')}\n"
                f"STORY_NAME: {story.get('STORY_NAME', 'Untitled Story')}\n"
                f"DESCRIPTION: {story.get('DESCRIPTION', 'No description provided.')}\n"
                f"ACCEPTANCE_CRITERIA:\n" +
                "\n".join([f"  - {ac}" for ac in story.get('ACCEPTANCE_CRITERIA', ["No acceptance criteria provided."])])
            )
            stories_for_prompt_list_chunk.append(story_text)
        
        formatted_user_stories_input_chunk = "\n\n---\nNext User Story:\n---\n\n".join(stories_for_prompt_list_chunk)

        user_content_chunk = f"""Lead Quality Assurance Architect AI,
Based on the following user stories (including their full descriptions and acceptance criteria), generate test cases. For each user story, develop {exact_test_cases_per_story} initial distinct conceptual test scenarios. If any single conceptual scenario is too complex to be exhaustively detailed in one test case object (especially regarding TEST_STEPS and EXPECTED_RESULTS for a novice tester), you MUST split it into multiple detailed test case objects using the `SPLIT_INFO` guidelines.

Adhere with **absolute and uncompromising precision** to all instructions, especially the **CRITICAL MANDATES FOR DETAILED TEST CASE FIELDS** and the **KEY FOCUS AREAS FOR EXTREME CLARITY**. The detail in `SETUP`, `TEST_STEPS` (with UI specifics and example data), and `EXPECTED_RESULTS` (exact messages, UI changes, data states) must be so thorough that a brand new tester can execute them without any questions.

Your entire response must be ONLY the specified JSON object.
USER STORIES INPUT : 
{formatted_user_stories_input_chunk}

Focus on providing **exhaustive, unambiguous, and exceptionally granular details**. The total number of test case objects per user story in the output array may exceed {exact_test_cases_per_story} if you determine splitting is necessary for comprehensive coverage and ultimate clarity of a complex concept.
\n\nRespond ONLY with a well-formed JSON array of test case objects. Do not include explanations or headers.

"""
        # messages_chunk = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content_chunk}]
        
        

        messages ={"role": "user", "parts": [{"text": user_content_chunk}]}

        chunk_test_cases_json_string = "" # Initialize for error handling scope
        try:
            print(f"--- Sending Chunk {chunk_number} to GPT-4o for Test Case Generation ---")

            model = genai.GenerativeModel(
                        model_name="gemini-2.5-pro",
                        generation_config=generation_config,
                        system_instruction=system_prompt
                        # If you had a system prompt, you would add it here:
                        # system_instruction="You are a helpful assistant designed to output JSON."
                    )
            # print("Messages",messages)
            response =  model.generate_content(contents=messages,generation_config={"response_mime_type": "application/json"})  # to receive structured JSON back)
                # response =  model.generate_content(messages=messages,)
                # response = client.chat.completions.create(
                #     model=AZURE_OPENAI_DEPLOYMENT_NAME,
                #     messages=messages_chunk,
                #     temperature=0,
                #     max_tokens=16000, # Ensure this is supported for OUTPUT by your model
                #     response_format={"type": "json_object"}
                # )
            # chunk_test_cases_json_string = response.choices[0].message.content
            chunk_test_cases_json_string = response.text.strip("```json").strip("```").strip()
            print("JSON Test Cases Results",chunk_test_cases_json_string)
            print(f"--- Chunk {chunk_number} Test Case JSON Received ---")
            # print(f"----------------------------------- Chunk {chunk_number} ----------------------------------- ")
            # print(chunk_test_cases_json_string)
            # print(f"----------------------------------- Chunk {chunk_number} ----------------------------------- ")            
            parsed_chunk_response = json.loads(chunk_test_cases_json_string)
            test_suites_from_chunk = parsed_chunk_response.get("all_test_suites")

            if test_suites_from_chunk is None:
                # Handle cases where "all_test_suites" key might be missing or response is unexpected
                error_msg = f"Test Case Generation for Chunk {chunk_number}: 'all_test_suites' key missing in API response. Response: {chunk_test_cases_json_string[:500]}..."
                print(f"Error: {error_msg}")
                return json.dumps({
                    "error": f"API response for test cases (Chunk {chunk_number}) missing 'all_test_suites' key.",
                    "details": f"Problem processing chunk {chunk_number} of {total_chunks}.",
                    "received_content_for_chunk": chunk_test_cases_json_string
                })
            
            all_generated_test_suites_concatenated.extend(test_suites_from_chunk)
            

        except json.JSONDecodeError as json_e:
            error_pos = json_e.pos
            context_before, context_after = 100, 100
            start_index = max(0, error_pos - context_before)
            end_index = min(len(json_e.doc), error_pos + context_after)
            problematic_snippet = json_e.doc[start_index:end_index]
            detailed_err_msg = (
                f"Test Case Generation for Chunk {chunk_number}: GPT-4o response was not valid JSON.\n"
                f"Error Type: JSONDecodeError. Message: {json_e.msg}\n"
                f"At Char: {error_pos}, Line: {json_e.lineno}, Col: {json_e.colno}\n"
                f"Snippet: ...{problematic_snippet}...\n"
                f"Start of chunk content: {chunk_test_cases_json_string[:200]}...\n"
                f"End of chunk content: ...{chunk_test_cases_json_string[-200:]}"
            )
            print(f"Error: {detailed_err_msg}")
            return json.dumps({
                "error": f"API response for test cases (Chunk {chunk_number}) was not valid JSON.",
                "details": {
                    "message": json_e.msg, "char_position": error_pos, "line_number": json_e.lineno,
                    "column_number": json_e.colno, "problematic_snippet": problematic_snippet,
                    "tip": "This often occurs if the API response was truncated due to exceeding max output tokens for the chunk, or if the generated content contains unescaped special characters."
                },
                "chunk_details": f"Problem processing chunk {chunk_number} of {total_chunks}.",
                "received_content_for_chunk": chunk_test_cases_json_string
            })
        except Exception as e:
            # Handling for other API errors for the current chunk
            response_text_snippet = "N/A"
            error_response = getattr(e, 'response', None)
            if error_response is not None:
                response_text_snippet = getattr(error_response, 'text', None)
                if response_text_snippet is None:
                    status_code = getattr(error_response, 'status_code', 'Unknown status')
                    response_text_snippet = f"Response object present but no text content. Status: {status_code}"
                else:
                    response_text_snippet = response_text_snippet[:200]
            
            err_detail = f"Error from Azure OpenAI for Chunk {chunk_number}: {type(e).__name__} - {e}. Snippet: {response_text_snippet}"
            print(err_detail)

            # Simplified context length check for chunk processing
            is_context_length_error = "context_length_exceeded" in str(e).lower() or \
                                      (hasattr(e, 'code') and str(getattr(e, 'code', '')).lower() == 'context_length_exceeded')
            
            if is_context_length_error:
                return json.dumps({
                    "error": f"Content for chunk {chunk_number} (stories {i+1}-{min(i+chunk_size, total_stories)}) is too extensive for the model's token limits.",
                    "details": str(e),
                    "chunk_details": f"Problem processing chunk {chunk_number} of {total_chunks} due to context length."
                })
            return json.dumps({
                "error": f"General API error during test case generation for chunk {chunk_number}.",
                "details": str(e),
                "chunk_details": f"Problem processing chunk {chunk_number} of {total_chunks}."
            })

    # After all chunks are processed successfully
    try:
        final_output_dict = {"all_test_suites": all_generated_test_suites_concatenated}
        print(f"\n--- All chunks processed. Attempting to create final concatenated Test Case JSON. Total suites: {len(all_generated_test_suites_concatenated)} ---")
        final_json_string = json.dumps(final_output_dict, indent=2)
        print(f"--- Final JSON string prepared (len: {len(final_json_string)}). Returning... ---")
        return final_json_string
    except Exception as e_final:
        # Catch any error during the final json.dumps, e.g., if concatenated list has non-serializable data (unlikely here but good practice)
        error_msg = f"CRITICAL ERROR: Failed to serialize the final concatenated test suites. Error: {type(e_final).__name__} - {e_final}"
        print(f"ERROR: {error_msg}")
        return json.dumps({
            "error": "Failed to create final JSON output from concatenated test suites.",
            "details": error_msg,
            "num_suites_collected": len(all_generated_test_suites_concatenated)
        })
    
# --- Main Processing Function---
def process_pdf_for_extraction_and_generation(pdf_path, main_output_dir="pdf_processing_outputs", test_cases_per_story=5):
    
    raw_text, img_paths, tables_content, us_generation_successful, us_json_str, us_json_path, us_err_path = [None]*7 # Initialize
    
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    review_out_path = os.path.join(main_output_dir, sanitize_filename(f"{pdf_basename}_review_output"))
    os.makedirs(review_out_path, exist_ok=True) # Ensure output path exists

    # Define paths for test case outputs
    tc_json_path = os.path.join(review_out_path, f"{sanitize_filename(pdf_basename)}_TestCases_SplitInfo.json")
    tc_err_path = os.path.join(review_out_path, f"{sanitize_filename(pdf_basename)}_TestCases_SplitInfo_Error.json")
        
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}"); return
    os.makedirs(main_output_dir, exist_ok=True)
 
    temp_dir = tempfile.mkdtemp()
    print(f"Temp dir: {temp_dir}")

    us_json_path = os.path.join(review_out_path, f"{sanitize_filename(pdf_basename)}_UserStories_MaxDetail.json") # Matching previous naming
    us_err_path = os.path.join(review_out_path, f"{sanitize_filename(pdf_basename)}_UserStories_MaxDetail_Error.json")
    
    try:
        raw_text, img_paths = extract_text_and_image_paths_from_pdf(pdf_path, temp_dir)
        tables_md = extract_tables_with_pdfplumber_to_markdown(pdf_path, chosen_strategy="lines")
        tables_content = "\n".join(tables_md) if tables_md else "No tables extracted from the PDF."
        
        save_review_file(main_output_dir, pdf_basename, raw_text, tables_md, img_paths) 

        print("\n--- User Story Generation Stage (Maximum Detail) ---")
        # Assuming generate_user_stories_with_gpt4o is the version from your previous "MAX_DETAIL" iteration
        us_json_str_generated = generate_user_stories_with_gpt4o(raw_text, tables_content, img_paths)


        print("\nUser Stories Generation Output (Maximum Detail):\n===========================================")
        parsed_us = None
        us_generation_successful_flag = False 
        try:
            parsed_us = json.loads(us_json_str_generated)
            if "error" in parsed_us:
                print(f"Error generating user stories (Max Detail): {parsed_us['error']}")
                if "details" in parsed_us: print(f"Details: {parsed_us['details']}")
                with open(us_err_path, "w", encoding="utf-8") as f: json.dump(parsed_us, f, indent=2)
                print(f"US (Max Detail) error details saved: {us_err_path}")
            elif "user_stories" not in parsed_us or not parsed_us["user_stories"]: 
                print("No user stories (Max Detail) were generated, or format unexpected.")
                with open(us_json_path, "w", encoding="utf-8") as f: json.dump(parsed_us, f, indent=2)
                print(f"US (Max Detail) output (empty/malformed) saved: {us_json_path}")
                with open(us_err_path, "w", encoding="utf-8") as f: json.dump({"error": "No user stories generated (Max Detail).", "response": parsed_us}, f, indent=2)
            else:
                
                with open(us_json_path, "w", encoding="utf-8") as f: json.dump(parsed_us, f, indent=2)
                print(f"User stories (Max Detail) saved: {us_json_path}")
                us_generation_successful_flag = True
                us_json_str_for_tc = us_json_str_generated 
        except json.JSONDecodeError:
            print(f"Invalid JSON from US generation (Max Detail): {us_json_str_generated[:1000]}...")
            with open(us_err_path, "w", encoding="utf-8") as f: f.write(us_json_str_generated) 
            us_json_str_for_tc = None # Cannot proceed with TC gen
        # --- End of User Story Generation part ---


        if us_generation_successful_flag and us_json_str_for_tc:
            print("\n--- Test Case Generation Stage (with Split Info) ---")
            # This now calls the modified test case generation function
            tc_json_str = generate_test_cases_for_user_stories_with_gpt4o(us_json_str_for_tc, exact_test_cases_per_story=test_cases_per_story)

            print("\nTest Cases Generation Output (with Split Info):\n===========================================")
            try:
                parsed_tc = json.loads(tc_json_str)
                if "error" in parsed_tc:
                    print(f"Error generating test cases (Split Info): {parsed_tc['error']}")
                    if "details" in parsed_tc: print(f"Details: {parsed_tc['details']}")
                    with open(tc_err_path, "w", encoding="utf-8") as f: json.dump(parsed_tc, f, indent=2)
                    print(f"TC (Split Info) error details saved: {tc_err_path}")
                elif "all_test_suites" not in parsed_tc or not parsed_tc["all_test_suites"]:
                    print("No test cases (Split Info) were generated, or format unexpected.")
                    with open(tc_json_path, "w", encoding="utf-8") as f: json.dump(parsed_tc, f, indent=2)
                    print(f"TC (Split Info) output (empty/malformed) saved: {tc_json_path}")
                    with open(tc_err_path, "w", encoding="utf-8") as f: json.dump({"error": "No test cases generated (Split Info).", "response": parsed_tc}, f, indent=2)
                else:
                    # print(json.dumps(parsed_tc, indent=2)) # Optionally print
                    with open(tc_json_path, "w", encoding="utf-8") as f: json.dump(parsed_tc, f, indent=2)
                    print(f"Test cases (with Split Info) saved: {tc_json_path}")
            except json.JSONDecodeError:
                print(f"Invalid JSON from TC generation (Split Info): {tc_json_str[:1000]}...")
                with open(tc_err_path, "w", encoding="utf-8") as f: f.write(tc_json_str)
                print(f"Raw TC (Split Info) error output saved: {tc_err_path}")
        else:
            print("\nSkipping test case generation (with Split Info): User story generation was not successful or yielded no stories.")

    finally:
        if os.path.exists(temp_dir): # temp_dir might not be defined if PDF path doesn't exist
            print(f"\nCleaning up temp dir: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error removing temp directory {temp_dir}: {e}")


if __name__ == "__main__":
    # pdf_file_path = "Docs/MakeMyTripPRD-Large.pdf" 
    pdf_file_path = "Docs/under-small-prd.pdf"
    if not os.path.exists(pdf_file_path):
        print(f"FATAL: PDF file not found at '{pdf_file_path}'.")
    else:
        
        output_directory = "pdf_ai_generated_artifacts_enhancet_test" 
        
        # Define how many detailed test cases you want per user story
        num_test_cases_per_story = 5 

        process_pdf_for_extraction_and_generation(
            pdf_file_path,
            main_output_dir=output_directory,
            test_cases_per_story=num_test_cases_per_story
        )
