import os
import sys
import codecs

# Nustatyti UTF-8 koduotę
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Pridėti projekto katalogą į Python kelią
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importuoti aplikaciją
from app import app

print(app.url_map)

if __name__ == '__main__':
    app.run(debug=True) 