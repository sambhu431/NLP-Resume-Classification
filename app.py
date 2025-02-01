
# Importing libraries

import streamlit as st
import pickle
import docx  
import PyPDF2  
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 

# Importing pickle files 
logReg_model = pickle.load(open('pickle_files/clfLR.pkl', 'rb'))  
tfidf = pickle.load(open('pickle_files/tfIDF.pkl', 'rb'))  
encoding = pickle.load(open('pickle_files/encoderLabel.pkl', 'rb'))  
STOPWORDS = set(stopwords.words('english'))

# Creating a function to clean input data
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # We can try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, we can try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
  
  corpus = []
  input_resume = cleanResume(input_resume)
  stemmer = PorterStemmer()
  review = re.sub('[^a-zA-Z]', ' ', input_resume)
  lowerReview = review.lower().split()
  review = [stemmer.stem(word) for word in lowerReview if not word in STOPWORDS]
  review = ' '.join(review)
  corpus.append(review)
  
  vectorized_text = tfidf.transform(corpus)

#  threshold = 0.078785
  threshold =  0.076788

  predicted_category = logReg_model.predict_proba(vectorized_text)
  max_proba = float(max(predicted_category[0]))

  if input_resume.strip() == "":
    return " Error: Input is empty. "
  elif max_proba < threshold and len(input_resume) < 200 :
    return " Error: The input might be invalid or less than minimum values with the following probability " , max_proba
  elif max_proba < threshold:
      return " Invalid with a probability count of ", max_proba
  else:
      vectorized_text = vectorized_text.toarray()
      predicted_category = logReg_model.predict(vectorized_text)
      predicted_category_name = encoding.inverse_transform(predicted_category)
      return predicted_category_name[0] , ' with a probability of ' , max_proba 






# Streamlit app 
def main():

    st.title("RESUME CATEGORIZATION")
    st.text("Please upload your resume in PDF, TXT, or DOCX format and the job category will get predicted.")
    st.markdown(f"<h3> Please restrain giving empty inputs as well as inputs less than 200 length </h3>",unsafe_allow_html=True)

    # File upload section
    uploaded_file = st.file_uploader("Please Upload Your Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        # Extract text from the uploaded file
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.markdown(f"<h3> The predicted Profession is: {category} </h3>",unsafe_allow_html=True)            

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
    
    userInput = st.text_area("Enter You Resume Details")
    if st.button("Submit"):
        try:
            category = pred(userInput)
            st.markdown(f"<h3> The predicted Profession  is : {category} </h3>",unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()





















