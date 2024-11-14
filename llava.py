import os
import pandas as pd
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

#Load the LLaVA processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)
model.to("cuda:0")
model.config.pad_token_id = model.config.eos_token_id  # Suppress warning

# Define paths
image_dir = 'tetris_photos/'  # Path to your images folder
csv_path = 'Dataset Meta Data - QA (1).csv'  # Path to your CSV file

# Define the system prompt
system_prompt = "You are a vision-language reasoning assistant. Answer the numerical question based on the provided image."

# Read the CSV file
questions_data = pd.read_csv(csv_path)

# Initialize results
results = []
max_questions = 3  # Change this to the desired number of questions

# Process each image and question
for idx, row in questions_data.iterrows():
    if idx >= max_questions:
        break

    # Generate the image prefix (e.g., '001')
    image_prefix = f"{int(row['Image']):03}"  # '001', '002', etc.

    # Find the file that starts with the correct prefix
    matching_files = [f for f in os.listdir(image_dir) if f.startswith(image_prefix)]
    if not matching_files:
        print(f"Image with prefix '{image_prefix}' not found in {image_dir}. Skipping...")
        results.append({
            "Image Prefix": image_prefix,
            "Question": row["Questions"],
            "Error": "Image not found",
        })
        continue

    image_path = os.path.join(image_dir, matching_files[0])  # Use the first match
    question = row["Questions"]
    correct_answer = int(row["Answers"])  # Correct answer from the dataset

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Define the conversation template for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{system_prompt}\n\nQuestion: {question}"},
                    {"type": "image"},
                ],
            },
        ]

        # Apply the chat template and process the image and prompt
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

        # Generate the response
        output = model.generate(**inputs, max_new_tokens=100)
        response = processor.decode(output[0], skip_special_tokens=True)

        # Clean up response to remove prompt echo and other tags
        clean_response = response.replace(prompt, "").replace("[INST]", "").replace("[/INST]", "").strip()
        first_number = int(clean_response.split()[0])  # Extract the first word as a number

        # Calculate absolute error and estimation type
        absolute_error = abs(first_number - correct_answer)
        estimation = (
            "Correct" if first_number == correct_answer else
            "Overestimated" if first_number > correct_answer else
            "Underestimated"
        )

        print(f"Image: {image_path}")
        print(f"Question: {question}")
        print(f"Model Response: {clean_response}")
        print(f"First Number: {first_number}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Absolute Error: {absolute_error}")
        print(f"Estimation: {estimation}\n")

        # Save result
        results.append({
            "Image": image_path,
            "Question": question,
            "Model Response": clean_response,
            "Extracted Number": first_number,
            "Correct Answer": correct_answer,
            "Absolute Error": absolute_error,
            "Estimation": estimation,
        })

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        results.append({
            "Image": image_path,
            "Question": question,
            "Error": str(e),
        })

# Save results to a new CSV file
output_df = pd.DataFrame(results)
output_file = "llava_results.csv"
output_df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
