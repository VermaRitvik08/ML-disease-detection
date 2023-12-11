mkdir -p ~/.streamlit/


echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "
[theme]
primaryColor=’#020202’
backgroundColor=’#c4c3c3’
secondaryBackgroundColor=’#ebd316’
font = ‘sans serif’
" > ~/.streamlit/config.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml


