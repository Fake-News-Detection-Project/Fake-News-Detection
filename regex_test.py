import re

pattern= r'([a-zA-Z/]{1,})'

text = "this is A 891 long TEXT 54894 contaiNinG 02555 LOT of words and 15".lower()

print(re.split(pattern, text))