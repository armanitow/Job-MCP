from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import json

# Initialize Claude client
def init_claude():
    return ChatAnthropic(
        api_key="your-claude-api-key", 
        model="claude-sonnet-4-20250514"
    )

def parse_resume(resume_text, client):
    """
    Parse a resume and extract structured information.
    
    Args:
        resume_text: The raw text of the resume
        client: Initialized ChatAnthropic client
    
    Returns:
        Dictionary containing parsed resume data
    """
    system_prompt = """You are a resume parsing expert. Extract structured information from resumes.
Return your response as valid JSON with the following structure:
{
  "personal_info": {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "portfolio": ""
  },
  "summary": "",
  "work_experience": [
    {
      "company": "",
      "position": "",
      "start_date": "",
      "end_date": "",
      "description": "",
      "achievements": []
    }
  ],
  "education": [
    {
      "institution": "",
      "degree": "",
      "field": "",
      "graduation_date": "",
      "gpa": ""
    }
  ],
  "skills": {
    "technical": [],
    "soft": [],
    "languages": []
  },
  "certifications": [],
  "projects": [
    {
      "name": "",
      "description": "",
      "technologies": []
    }
  ]
}

Extract all available information. Use empty strings or empty arrays if information is not present."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please parse this resume:\n\n{resume_text}")
    ]
    
    response = client.invoke(messages)
    
    try:
        # Extract JSON from response
        content = response.content
        # Find JSON in the response (it might be wrapped in markdown)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            json_str = content[json_start:json_end].strip()
        else:
            json_str = content
        
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse response: {e}", "raw_response": response.content}

def chatbot_interface():
    """Simple chatbot interface for resume parsing."""
    print("=== Resume Parser Chatbot ===")
    print("Paste your resume text below. Type 'END' on a new line when finished.\n")
    
    # Collect resume text
    resume_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        resume_lines.append(line)
    
    resume_text = '\n'.join(resume_lines)
    
    if not resume_text.strip():
        print("No resume text provided!")
        return
    
    print("\n🔄 Parsing resume...\n")
    
    # Initialize client and parse
    client = init_claude()
    parsed_data = parse_resume(resume_text, client)
    
    # Display results
    print("=" * 50)
    print("PARSED RESUME DATA")
    print("=" * 50)
    print(json.dumps(parsed_data, indent=2))
    
    return parsed_data

def parse_resume_file(file_path):
    """Parse resume from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        resume_text = f.read()
    
    client = init_claude()
    return parse_resume(resume_text, client)

# Example usage
if __name__ == "__main__":
    # Interactive chatbot mode
    chatbot_interface()
    
    # Or parse from file:
    # parsed_data = parse_resume_file("resume.txt")
    # print(json.dumps(parsed_data, indent=2))