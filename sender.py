import requests

url = 'http://10.213.111.106:5000/search-images'

txt = input().encode('utf-8')
response = requests.post(url, data = txt)
print(response)
from PIL import Image
from io import BytesIO

i = Image.open(BytesIO(response.content))
import matplotlib.pyplot
matplotlib.pyplot.imshow(i)
matplotlib.pyplot.show()
