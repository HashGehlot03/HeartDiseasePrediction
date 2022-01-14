mkdir -p ~/.streamlit/
echo "[general]
email = \" harishgcmc19@svvv.edu.in\"" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false" >> ~/.streamlit/config.toml
"
