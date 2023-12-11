mkdir -p ~/.streamlit/
echo "
[general]\n\
email = \"your-email@domain.com\"\n\
[theme]
primaryColor=’#020202’
backgroundColor=’#c4c3c3’
secondaryBackgroundColor=’#ebd316’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

