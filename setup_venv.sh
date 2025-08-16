rm -r rl_venv # Remove the venv if it exists.
python3 -m venv rl_venv # Create the venv
source rl_venv/bin/activate # Activate the venv
echo "You are now in a venv located at: $VIRTUAL_ENV" # Display the name of the virtual env
python3 -m pip install -r requirements.txt -q # Download all required libraries in the venv
echo "All requirements installed."