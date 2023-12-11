mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = ‘#84a3a7’
backgroundColor = ‘#EFEDE8’
secondaryBackgroundColor = ‘#fafafa’
textColor= ‘#424242’
font = ‘sans serif’
" > ~/.streamlit/config.toml

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\



" > ~/.streamlit/config.toml
