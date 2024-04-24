from langchain.prompts import PromptTemplate

routing_prompt_web_search = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on spanish BOE documents. \n
    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

routing_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or to a question processing tool. \n
    Use the vectorstore for questions related on spanish BOE documents. Use this list of topics that you can typically find in the BOE:
    Legislation: New laws passed by the Spanish Parliament, including national and regional legislation.
    Regulations and Decrees: Detailed rules and regulations for implementing laws, including Royal Decrees.
    Government Announcements: Official announcements and communications from various government departments and ministries.
    Legal Notices: Information on legal processes, including notifications of estate claims, summons, and other legal advertisements.
    Public Employment: Announcements about public sector job openings, civil service exams, and appointments.
    Judicial Appointments and Decisions: Information on appointments to the judiciary and summaries of significant judicial decisions.
    Economic and Financial Information: Data related to the national budget, public debt issues, and financial resolutions.
    Grants and Subsidies: Information about government grants and subsidies available to individuals, businesses, and organizations.
    Sanctions: Notices about penalties imposed by government bodies for various offenses.
    International Treaties: Treaties and agreements between Spain and other countries.
    Intellectual Property Registrations: Announcements related to patents, trademarks, and copyrights.
    Company and Business Regulations: Changes in commercial law that affect companies operating in Spain.
    Awards and Honors: Official listings of state awards and honors bestowed upon individuals and organizations. \n
    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use the question processing tool. Give a binary choice 'question_tool' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. 
    Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
