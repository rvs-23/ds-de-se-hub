# import streamlit as st
# import requests
# import time

# # Function to make API call based on method (GET or POST)
# def make_api_call(method, message_id):
#     if method == "GET":
#         # Simulate API call delay
#         time.sleep(3)
#         # Placeholder for GET request, replace with actual API call
#         return f"Results for GET request with message ID {message_id}"
#     elif method == "POST":
#         # Simulate API call delay
#         time.sleep(5)
#         # Placeholder for POST request, replace with actual API call
#         return f"Results for POST request with message ID {message_id}"

# # Set page title and favicon
# st.set_page_config(page_title="The GREEN Box", page_icon=":green_square:")

# # Page title
# st.title("The GREEN Box")

# # Sidebar title and description
# st.sidebar.title("Connect to System")
# st.sidebar.write("Make a POST or GET call to the system")

# # Dropdown for selecting API method
# method = st.sidebar.selectbox("Select Method", ["POST", "GET"])

# # Text input for message ID
# message_id = st.sidebar.text_input("Enter Message ID")

# # Button to initiate API call
# if st.sidebar.button("Fetch Results"):
#     # Show progress bar
#     progress_bar = st.progress(0)
#     for i in range(101):
#         time.sleep(0.05)
#         progress_bar.progress(i)
#     # Call API and display results
#     result = make_api_call(method, message_id)
#     st.success("Results Fetched!")
#     st.write(result)

# import streamlit as st
# import time

# # Function to simulate API call
# def simulate_api_call(message_id):
#     # Simulating API call delay
#     progress_bar = st.progress(0)
#     for i in range(101):
#         time.sleep(0.05)
#         progress_bar.progress(i)
#     # Simulating response
#     if message_id == "123":
#         return "Results for message ID 123: Lorem ipsum dolor sit amet, consectetur adipiscing elit."
#     elif message_id == "456":
#         return "Results for message ID 456: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
#     else:
#         return "No results found for the provided message ID."

# # Main function for Streamlit app
# def main():
#     st.title("The GREEN Box")

#     # Dropdown for selecting API call type
#     api_call_type = st.selectbox("Select API Call Type", ["POST", "GET"])

#     # Text input for entering message ID
#     message_id = st.text_input("Enter Message ID")

#     # Button to trigger API call
#     if st.button("Fetch Results"):
#         if message_id:
#             st.write("Fetching results...")
#             results = simulate_api_call(message_id)
#             st.success("Results fetched successfully!")
#             st.write(results)
#         else:
#             st.warning("Please enter a valid Message ID.")

# # Run the app
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import time

# # Function to simulate API call
# def simulate_api_call(message_ids):
#     # Simulating API call delay
#     progress_bar = st.progress(0)
#     for i in range(101):
#         time.sleep(0.05)
#         progress_bar.progress(i)
#     # Simulating response
#     results = {}
#     for message_id in message_ids:
#         if message_id == "123":
#             results[message_id] = "Results for message ID 123: Lorem ipsum dolor sit amet, consectetur adipiscing elit."
#         elif message_id == "456":
#             results[message_id] = "Results for message ID 456: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
#         else:
#             results[message_id] = "No results found for message ID " + message_id
#     return results

# # Main function for Streamlit app
# def main():
#     st.title("The GREEN Box")
#     st.write("Welcome to The GREEN Box, Kams! This app allows you to fetch results from Stargate using API calls.")

#     # Sidebar
#     st.sidebar.subheader("Settings")
#     api_call_type = st.sidebar.selectbox("Select API Call Type", ["POST", "GET"])

#     # Main content
#     st.write("## Enter Message ID(s)")
#     message_ids = st.text_input("Enter Message ID(s) separated by commas (,)", "123,456").split(",")

#     # Button to trigger API call
#     if st.button("Fetch Results"):
#         if any(message_ids):
#             st.write("Fetching results...")
#             results = simulate_api_call(message_ids)
#             st.success("Results fetched successfully!")

#             # Display results
#             st.write("## Results")
#             for message_id, result in results.items():
#                 st.write(f"### Message ID: {message_id}")
#                 st.write(result)
#         else:
#             st.warning("Please enter at least one Message ID.")

# # Run the app
# if __name__ == "__main__":
#     main()



import streamlit as st
import time

# Function to simulate API call
def simulate_api_call(message_ids):
    # Simulating API call delay
    progress_bar = st.progress(0)
    for i in range(101):
        time.sleep(0.05)
        progress_bar.progress(i)
    # Simulating response
    results = {}
    for message_id in message_ids:
        if message_id == "123":
            results[message_id] = "Results for message ID 123: Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        elif message_id == "456":
            results[message_id] = "Results for message ID 456: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        else:
            results[message_id] = "No results found for message ID " + message_id
    return results

# Main function for Streamlit app
def main():
    st.title("The GREEN Box")
    st.write("Welcome to The GREEN Box! This app allows you to fetch results from an external system using API calls.")

    # Sidebar
    st.sidebar.title("Settings")
    api_call_type = st.sidebar.selectbox("Select API Call Type", ["POST", "GET"])

    # Main content
    st.write("## Enter Message ID(s)")
    message_ids = st.text_input("Enter Message ID(s) separated by commas (,)", "123,456").split(",")

    # Button to trigger API call
    if st.button("Fetch Results"):
        if any(message_ids):
            st.write("Fetching results...")
            with st.spinner("Fetching results..."):
                results = simulate_api_call(message_ids)
                st.success("Results fetched successfully!")

                # Display results
                st.write("## Results")
                st.json(results)  # Output formatted as JSON
        else:
            st.warning("Please enter at least one Message ID.")

# Run the app
if __name__ == "__main__":
    main()

