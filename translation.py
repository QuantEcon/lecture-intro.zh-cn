import openai


def split_text(content, chunk_size=3000):
    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size

        # If we are at the end of the content, just append the rest
        if end >= len(content):
            chunks.append(content[start:])
            break
        
        # Check if the end is in the middle of a line
        if content[end] != '\n':
            # Find the nearest line separator after the chunk size
            next_line_break = content.find('\n', end)
            if next_line_break == -1:
                # If no further line breaks are found, just append the rest
                chunks.append(content[start:])
                break
            else:
                end = next_line_break + 1
        
        chunks.append(content[start:end].strip())
        start = end

    return chunks

def translate_cn(input_file, assistant_id):
    # Initialize the OpenAI client
    client = openai.OpenAI()

    # Read the content of the input markdown file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into chunks (e.g., 3000 characters each)
    chunks = split_text(content, chunk_size=3000)

    # Initialize the OpenAI client and thread
    thread = client.beta.threads.create()

    translated_content = ""

    for chunk in chunks:
        # Create and poll the run for each chunk
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_cn_id,
            instructions="Please translate the following content into simplified Chinese. Give the results directly: " + chunk
        )
        
        if run.status == 'completed': 
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            translated_content += messages.data[0].content[0].text.value
            print(translated_content)
        else:
            print(f"Translation failed for chunk: {chunk[:50]}... Status: {run.status}")
            continue

    # Create the output file name
    output_file = input_file.replace('.md', '_cn.md')

    # Write the translated content to the new markdown file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translated_content)

    print(f"Translated content has been saved to {output_file}")

if __name__ == "__main__":
    input_file = "lectures/ar1_processes.md"
    print(openai.beta.assistants.list())
    assistant_cn_id = 'asst_kWnZVKQEHRY1Db6ezdbKnRIy'
    translate_cn(input_file, assistant_cn_id)