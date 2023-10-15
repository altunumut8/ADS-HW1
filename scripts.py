# These are the submitted solutions for hackerrank python problems.

# Say Hello World With Python.
import numpy as np
import numpy
import operator
import xml.etree.ElementTree as etree
from html.parser import HTMLParser
import email.utils
from datetime import datetime
import calendar
from collections import deque
from collections import OrderedDict
from collections import namedtuple
from collections import defaultdict
from collections import Counter
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    n = int(input().strip())

    if n % 2 != 0:
        print('Weird')

    elif 2 <= n <= 5:
        print('Not Weird')

    elif 6 <= n <= 20:
        print('Weird')

    elif n > 20:
        print('Not Weird')


# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a+b)
    print(a-b)
    print(a*b)


# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a//b)
    print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())

    for i in range(0, n):
        print(i**2)

# Write a function


def is_leap(year):
    leap = False

    # Write your logic here
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        leap = True

    return leap


# Print Function
if __name__ == '__main__':
    n = int(input())

    number_list = [str(i) for i in range(1, n+1)]
    number_string = ''.join(number_list)
    print(number_string)

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    output = [[i, j, k] for i in range(
        x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]

    print(output)


# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    unique_numbers = list(set(arr))

    max_number = max(unique_numbers)

    unique_numbers.remove(max_number)

    runner_up = max(unique_numbers)

    print(runner_up)

# Nested Lists
if __name__ == '__main__':

    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])

    students.sort(key=lambda x: (x[1], x[0]))

    second_lowest_grade = None
    for student in students:
        if second_lowest_grade is None:
            second_lowest_grade = student[1]
        elif student[1] > second_lowest_grade:
            second_lowest_grade = student[1]
            break
    second_lowest_students = [student[0]
                              for student in students if student[1] == second_lowest_grade]
    second_lowest_students.sort()
    for name in second_lowest_students:
        print(name)


# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    if query_name in student_marks:
        marks_list = student_marks[query_name]
        average = sum(marks_list) / len(marks_list)
        average = "{:.2f}".format(average)
        print(average)

    else:
        print('Not found')


# Lists
if __name__ == '__main__':
    N = int(input())
    my_list = []

    for _ in range(N):
        operation = input().split()
        cmd = operation[0]

        if cmd == "insert":
            i = int(operation[1])
            e = int(operation[2])
            my_list.insert(i, e)
        elif cmd == "print":
            print(my_list)
        elif cmd == "remove":
            e = int(operation[1])
            my_list.remove(e)
        elif cmd == "append":
            e = int(operation[1])
            my_list.append(e)
        elif cmd == "sort":
            my_list.sort()
        elif cmd == "pop":
            my_list.pop()
        elif cmd == "reverse":
            my_list.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    my_tuple = tuple(integer_list)
    print(hash(my_tuple))

# sWAP cASE


def swap_case(s):
    swapped_string = ""
    for char in s:
        if char.islower():
            swapped_string += char.upper()
        elif char.isupper():
            swapped_string += char.lower()
        else:
            swapped_string += char
    return swapped_string


# String Split and Join
def split_and_join(line):
    # write your code here
    split_string = line.split(" ")
    result_string = "-".join(split_string)

    return result_string


if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#


def print_full_name(first_name, last_name):
    # Write your code here
    message = f"Hello {first_name} {last_name}! You just delved into python."
    print(message)


# Mutations
def mutate_string(string, position, character):
    modified_string = string[:position] + character + string[position + 1:]
    return modified_string


# Find a string
def count_substring(string, sub_string):
    count = 0
    index = 0

    while index < len(string):
        index = string.find(sub_string, index)
        if index == -1:
            break
        count += 1
        index += 1

    return count


# String Validators
if __name__ == '__main__':
    s = input()

    def has_alphanumeric_characters(s):
        return any(c.isalnum() for c in s)

    def has_alpha_characters(s):
        return any(c.isalpha() for c in s)

    def has_digit_characters(s):
        return any(c.isdigit() for c in s)

    def has_lowercase_characters(s):
        return any(c.islower() for c in s)

    def has_uppercase_characters(s):
        return any(c.isupper() for c in s)

    print(has_alphanumeric_characters(s))
    print(has_alpha_characters(s))
    print(has_digit_characters(s))
    print(has_lowercase_characters(s))
    print(has_uppercase_characters(s))


# Text Alignment
# Enter your code here. Read input from STDIN. Print output to STDOUT
def print_hackerrank_logo(thickness):
    c = 'H'
    # Top side
    for i in range(thickness):
        print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

    # Second level
    for i in range(thickness + 1):
        print((c * thickness).center(thickness * 2) +
              (c * thickness).center(thickness * 6))

    # Middle level
    for i in range((thickness + 1) // 2):
        print((c * thickness * 5).center(thickness * 6))

    # Third level
    for i in range(thickness + 1):
        print((c * thickness).center(thickness * 2) +
              (c * thickness).center(thickness * 6))

    # Bottom side
    for i in range(thickness):
        print(((c * (thickness - i - 1)).rjust(thickness) + c +
              (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))


if __name__ == '__main__':
    thickness = int(input())
    print_hackerrank_logo(thickness)

# Text Wrap


def wrap(string, max_width):
    parts = [string[i:i+max_width] for i in range(0, len(string), max_width)]
    return '\n'.join(parts)

# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT


def print_design(rows, columns):
    pattern = [('.|.' * (2 * i + 1)).center(columns, '-')
               for i in range(rows // 2)]
    welcome = 'WELCOME'.center(columns, '-')
    mat = '\n'.join(pattern + [welcome] + pattern[::-1])
    print(mat)


rows, columns = map(int, input().split())

print_design(rows, columns)

# String Formatting


def print_formatted(number):
    width = len(bin(number)[2:])
    for i in range(1, number + 1):
        decimal_str = str(i).rjust(width)
        octal_str = oct(i)[2:].rjust(width)
        hexadecimal_str = hex(i)[2:].upper().rjust(width)
        binary_str = bin(i)[2:].rjust(width)
        print(f"{decimal_str} {octal_str} {hexadecimal_str} {binary_str}")

# Alphabet Rangoli


def print_rangoli(n):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    width = 4*n - 3
    result = []

    for i in range(n):
        line = "-".join(alphabets[n-1:i:-1] + alphabets[i:n])
        result.append(line.center(width, "-"))

    print("\n".join(result[:0:-1] + result))


# Capitalize!
# Complete the solve function below.
def solve(s):
    return ' '.join(word.capitalize() for word in s.split(' '))

# The Minion Game


def minion_game(s):
    vowels = 'AEIOU'
    stuart_score = 0
    kevin_score = 0
    for i in range(len(s)):
        if s[i] in vowels:
            kevin_score += (len(s)-i)
        else:
            stuart_score += (len(s)-i)
    if kevin_score > stuart_score:
        print("Kevin", kevin_score)
    elif kevin_score < stuart_score:
        print("Stuart", stuart_score)
    else:
        print("Draw")

# Merge the Tools!


def merge_the_tools(s, k):
    sub_strings = [s[i:i+k] for i in range(0, len(s), k)]
    for sub in sub_strings:
        print("".join(sorted(set(sub), key=sub.index)))


if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# Introduction to Sets


def average(arr):
    distinct_nums = set(arr)
    average = sum(distinct_nums) / len(distinct_nums)
    return average

# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT


def print_symmetric_difference(a, b):
    sym_diff = sorted(list(a.symmetric_difference(b)))
    for num in sym_diff:
        print(num)


m = int(input())
set_a = set(map(int, input().split()))
n = int(input())
set_b = set(map(int, input().split()))

print_symmetric_difference(set_a, set_b)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
stamps = set()

for _ in range(n):
    stamps.add(input())

print(len(stamps))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
num_cmds = int(input())

for _ in range(num_cmds):
    cmd = input().split()
    try:
        if cmd[0] == "pop":
            s.pop()
        elif cmd[0] == "remove":
            s.remove(int(cmd[1]))
        else:
            s.discard(int(cmd[1]))
    except KeyError:
        continue

print(sum(s))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
english_subs = set(map(int, input().split()))
m = int(input())
french_subs = set(map(int, input().split()))
total_subs = len(english_subs.union(french_subs))

print(total_subs)


# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
english = set(map(int, input().split()))
m = int(input())
french = set(map(int, input().split()))
print(len(english.intersection(french)))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
english = set(map(int, input().split()))
m = int(input())
french = set(map(int, input().split()))
print(len(english.difference(french)))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
english = set(map(int, input().split()))
m = int(input())
french = set(map(int, input().split()))
print(len(english.symmetric_difference(french)))

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
s = set(map(int, input().split()))
ops = int(input())

for _ in range(ops):
    cmd, _ = input().split()
    other_set = set(map(int, input().split()))
    if cmd == 'intersection_update':
        s.intersection_update(other_set)
    elif cmd == 'update':
        s.update(other_set)
    elif cmd == 'symmetric_difference_update':
        s.symmetric_difference_update(other_set)
    elif cmd == 'difference_update':
        s.difference_update(other_set)

print(sum(s))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
k = int(input())
room_numbers = list(map(int, input().split()))
unique_rooms = set(room_numbers)
captains_room = (sum(unique_rooms) * k - sum(room_numbers)) // (k - 1)
print(captains_room)

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
t = int(input())

for _ in range(t):
    a = int(input())
    set_a = set(map(int, input().split()))
    b = int(input())
    set_b = set(map(int, input().split()))

    if set_a.issubset(set_b):
        print("True")
    else:
        print("False")

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
A = set(map(int, input().split()))
n = int(input())
superset = True

for _ in range(n):
    B = set(map(int, input().split()))
    if not (A.issuperset(B) and (len(A) > len(B))):
        superset = False
        break

print(superset)

# No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = map(int, input().split())
array = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
happiness = 0
for i in array:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1
print(happiness)

# Collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
shoes = Counter(map(int, input().split()))
nc = int(input())

income = 0
for _ in range(nc):
    size, price = map(int, input().split())
    if shoes[size]:
        income += price
        shoes[size] -= 1

print(income)

# DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m = map(int, input().split())
A = defaultdict(list)
for i in range(1, n+1):
    A[input()].append(str(i))

for _ in range(m):
    print(' '.join(A[input()]) or '-1')

# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT

num_of_students = int(input())
fields = list(input().split())
sum_of_marks = 0
Student = namedtuple('Student', fields)

for _ in range(num_of_students):
    ID, MARKS, NAME, CLASS = input().split()
    student = Student(ID, MARKS, NAME, CLASS)
    sum_of_marks += int(student.MARKS)

avg_of_marks = sum_of_marks / num_of_students

print("%.2f" % avg_of_marks)

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
sold_items = OrderedDict()
for _ in range(n):
    item, space, quantity = input().rpartition(' ')
    sold_items[item] = sold_items.get(item, 0) + int(quantity)

for item, quantity in sold_items.items():
    print(item, quantity)

# Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
word_order = OrderedDict()

for _ in range(n):
    word = input()
    word_order[word] = word_order.get(word, 0) + 1

print(len(word_order))
print(*word_order.values())


# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT

N = int(input())
d = deque()

for _ in range(N):
    operation = input().split()

    if operation[0] == 'append':
        d.append(int(operation[1]))
    elif operation[0] == 'appendleft':
        d.appendleft(int(operation[1]))
    elif operation[0] == 'pop':
        d.pop()
    elif operation[0] == 'popleft':
        d.popleft()

for i in d:
    print(i, end=' ')


# Company Logo
#!/bin/python3


if __name__ == '__main__':
    s = input().strip()

    counter = Counter(s)
    counter_sorted = sorted(
        counter.items(), key=lambda pair: (-pair[1], pair[0]))
    for char, freq in counter_sorted[:3]:
        print(f'{char} {freq}')


# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT

T = int(input())

for _ in range(T):
    n = int(input())
    cubes = deque(map(int, input().split()))
    topCube = max(cubes[0], cubes[-1])

    while cubes:
        if cubes[0] >= cubes[-1] and topCube >= cubes[0]:
            topCube = cubes.popleft()
        elif topCube >= cubes[-1]:
            topCube = cubes.pop()
        else:
            print('No')
            break
    else:
        print('Yes')


# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
date = input().split()
weekday_number = calendar.weekday(int(date[2]), int(date[0]), int(date[1]))
weekday_name = calendar.day_name[weekday_number].upper()
print(weekday_name)


# Time Delta
#!/bin/python3
# Complete the time_delta function below.


def time_delta(t1, t2):
    time_format = "%a %d %b %Y %H:%M:%S %z"

    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)

    return str(int(abs((t1-t2).total_seconds())))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input().strip())

for _ in range(T):
    a, b = input().strip().split()

    try:
        print(int(a) // int(b))
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as e:
        print("Error Code:", e)

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, X = map(int, input().split())
marks = [map(float, input().split()) for _ in range(X)]
students = list(zip(*marks))
for i in students:
    print(sum(i)/len(i))


# Athlete Sort
#!/bin/python3


if __name__ == '__main__':
    N, M = map(int, input().split())
    table = [list(map(int, input().split())) for _ in range(N)]
    k = int(input())
    sorted_table = sorted(table, key=lambda row: row[k])
    for row in sorted_table:
        print(' '.join(map(str, row)))

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
s = input()

lowercase = sorted([c for c in s if c.islower()])
uppercase = sorted([c for c in s if c.isupper()])
odds = sorted([c for c in s if c.isdigit() and int(c) % 2 != 0])
evens = sorted([c for c in s if c.isdigit() and int(c) % 2 == 0])

print(''.join(lowercase + uppercase + odds + evens))

# Map and Lambda Function


def cube(x): return x**3


def fibonacci(n):
    fib_list = [0, 1]
    while len(fib_list) < n:
        fib_list.append(fib_list[-1] + fib_list[-2])
    return fib_list[:n]


# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT


def is_float(num):
    pattern = r'^[-+]?[0-9]*\.[0-9]+$'
    return bool(re.match(pattern, num))


if __name__ == "__main__":
    n = int(input())
    for _ in range(n):
        num = input().strip()
        print(is_float(num))

# Group(), Groups() & Groupdict()

s = input().strip()

regex_pattern = r'.*?([a-zA-Z0-9])\1.*?'

match = re.search(regex_pattern, s)

if match:
    print(match.group(1))
else:
    print("-1")

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT

vowels = 'aeiouAEIOU'
consonants = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'

regex_pattern = r"(?<=[" + consonants + "])([" + \
    vowels + "]{2,})(?=[" + consonants + "])"

matches = re.findall(regex_pattern, input().strip())

if len(matches) == 0:
    print("-1")
else:
    for match in matches:
        print(match)

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT

s = input().strip()
k = input().strip()

pattern = re.compile(r'(?={})'.format(k))

matches = pattern.finditer(s)
found = False

for match in matches:
    print((match.start(), match.start() + len(k) - 1))
    found = True

if not found:
    print((-1, -1))


# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())

for _ in range(n):
    s = input()
    s = re.sub(r'(?<= )&&(?= )', 'and', s)
    s = re.sub(r'(?<= )\|\|(?= )', 'or', s)
    print(s)


# Validating Roman Numerals
regex_pattern = '^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
print(str(bool(re.match(regex_pattern, input()))))


# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT

n = int(input())
for _ in range(n):
    phone_number = input().strip()
    match = re.match(r'^[7|8|9]\d{9}$', phone_number)
    if match:
        print('YES')
    else:
        print('NO')

# Validating and Parsing Email Addresses

n = int(input())
for _ in range(n):
    parsed_email = email.utils.parseaddr(input())
    email_address = parsed_email[1]
    match = re.match(
        r'^[a-zA-Z][\w\_\-\.]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$', email_address)
    if match:
        print(email.utils.formataddr(parsed_email))

# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT

for _ in range(int(input())):
    matches = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if matches:
        print(*matches, sep='\n')


# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT

# Custom HTML Parser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for attr in attrs:
            print('-> {} > {}'.format(attr[0], attr[1]))

    def handle_endtag(self, tag):
        print('End   :', tag)

    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for attr in attrs:
            print('-> {} > {}'.format(attr[0], attr[1]))


parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())


# HTML Parser - Part 2
# Enter your code here. Read input from STDIN. Print output to STDOUT


class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        number_of_line = len(data.split('\n'))
        if number_of_line > 1:
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        if data.strip():
            print(data)

    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)


html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'

parser = MyHTMLParser()
parser.feed(html)


# Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT


class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]


html = '\n'.join([input() for _ in range(int(input()))])
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT

for _ in range(int(input())):
    uid = input().strip()
    uid = ''.join(sorted(uid))

    if (re.search(r'[A-Z]{2}', uid) and  # 2 uppercase alphabets
        re.search(r'\d{3}', uid) and  # 3+ digits
        not re.search(r'[^A-Za-z0-9]', uid) and  # only alphanumeric
        not re.search(r'(\w)\1', uid) and  # no repetition
            len(uid) == 10):  # exactly 10 characters
        print('Valid')
    else:
        print('Invalid')


# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT

for _ in range(int(input())):
    card_number = input().strip()
    match = re.search(r"^[456]([\d]{15}|[\d]{3}(-[\d]{4}){3})$",
                      card_number) and not re.search(r"([\d])\1\1\1", card_number.replace("-", ""))
    if match:
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^[1-9][\d]{5}$"    # Do not delete 'r'.
# Do not delete 'r'..
regex_alternating_repetitive_digit_pair = r"(?=(\d)\d\1)"


P = input()

print(bool(re.match(regex_integer_in_range, P))
      and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

# Matrix Script
#!/bin/python3


n, m = map(int, input().split())
matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

matrix_script = "".join([matrix[j][i] for i in range(m) for j in range(n)])

decoded_script = re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", matrix_script)
print(decoded_script)


# Re.split()
regex_pattern = r"[.,]"  # Do not delete 'r'.

print("\n".join(re.split(regex_pattern, input())))


# XML 1 - Find the Score


def get_attr_number(node):
    return sum(len(elem.items()) for elem in node.iter())


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


# XML2 - Find the Maximum Depth

maxdepth = 0


def depth(elem, level):
    global maxdepth
    # increment level by 1
    level += 1

    # update max_depth if level is higher
    if maxdepth < level:
        maxdepth = level

    # recursively call depth on each child
    for child in elem:
        depth(child, level)


if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml = xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)


# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # standardize the numbers
        l = ['+91 {} {}'.format(n[-10:-5], n[-5:]) for n in l]
        # call the decorated function
        return f(l)
    return fun


@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')


if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)


# Decorators 2 - Name Directory


def person_lister(f):
    def inner(people):
        # Sort the list by age
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner


@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]


if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# Arrays


def arrays(arr):
    numpy_arr = numpy.array(arr, float)
    return numpy_arr[::-1]


arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and Reshape
# Enter your code here. Read input from STDIN. Print output to STDOUT

numbers = list(map(int, input().split()))
array = numpy.array(numbers)

reshaped_array = numpy.reshape(array, (3, 3))
print(reshaped_array)


# Transpose and Flatten
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = map(int, input().split())
array = np.array([input().strip().split() for _ in range(n)], int)
print(array.transpose())
print(array.flatten())


# Concatenate
# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m, p = map(int, input().split())
array1 = np.array([input().split() for _ in range(n)], int)
array2 = np.array([input().split() for _ in range(m)], int)

print(np.concatenate((array1, array2), axis=0))


# Zeros and Ones
# Enter your code here. Read input from STDIN. Print output to STDOUT,

dims = tuple(map(int, input().split()))

print(numpy.zeros(dims, dtype=numpy.int))
print(numpy.ones(dims, dtype=numpy.int))

# Eye and Identity
# Enter your code here. Read input from STDIN. Print output to STDOUT
np.set_printoptions(sign=' ')

n, m = map(int, input().split())
print(np.eye(n, m, k=0))


# Array Mathematics
# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m = map(int, input().split())
a, b = (numpy.array([input().split()
        for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a % b, a**b, sep='\n')

# Floor, Ceil and Rint
# Enter your code here. Read input from STDIN. Print output to STDOUT

np.set_printoptions(sign=' ')

array = np.array(input().split(), float)

print(np.floor(array))
print(np.ceil(array))
print(np.rint(array))


# Sum and Prod
# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m = map(int, input().split())

arr = []
for _ in range(n):
    arr.append(list(map(int, input().split())))

arr = np.array(arr)
sum_arr = np.sum(arr, axis=0)
product = np.prod(sum_arr)

print(product)


# Min and Max
# Enter your code here. Read input from STDIN. Print output to STDOUT

n, m = map(int, input().split())

arr = np.array([input().split() for _ in range(n)], int)

minimum = np.min(arr, axis=1)
maximum = np.max(minimum)

print(maximum)


# Mean, Var, and Std
# Enter your code here. Read input from STDIN. Print output to STDOUT

N, M = map(int, input().split())

arr = np.array([list(map(int, input().split())) for _ in range(N)])

print(np.mean(arr, axis=1))
print(np.var(arr, axis=0))
print(np.std(arr).round(11))

# Dot and Cross

n = int(input())

arr1 = np.array([input().split() for _ in range(n)], int)
arr2 = np.array([input().split() for _ in range(n)], int)

dot_product = np.dot(arr1, arr2)
print(dot_product)

# Inner and Outer
# Enter your code here. Read input from STDIN. Print output to STDOUT

arr1 = np.array(input().split(), int)
arr2 = np.array(input().split(), int)

inner_product = np.inner(arr1, arr2)
outer_product = np.outer(arr1, arr2)

print(inner_product)
print(outer_product)

# Polynomials
# Enter your code here. Read input from STDIN. Print output to STDOUT

coefficients_P = list(map(float, input().split()))

x = float(input())

result = np.polyval(coefficients_P, x)
print(result)


# Linear Algebra
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
matrix = np.array([input().split() for _ in range(n)], float)
determinant = np.linalg.det(matrix)
print(round(determinant, 2))


# Triangle Quest
# Enter your code here. Read input from STDIN. Print output to STDOUT
for i in range(1, int(input())):
    print((10**i-1)//9*i)


# Birthday Cake Candles
#!/bin/python3


#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#


def birthdayCakeCandles(candles):
    # Write your code here
    max_height = max(candles)
    return candles.count(max_height)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps
#!/bin/python3


#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#


def kangaroo(x1, v1, x2, v2):
    if x1 < x2 and v1 > v2 and (x2 - x1) % (v1 - v2) == 0:
        return 'YES'
    else:
        return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising
#!/bin/python3


#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#


def viralAdvertising(n):
    # Write Your Code Here
    shared = 5
    cumulative = 0
    for _ in range(n):
        liked = shared // 2
        shared = liked * 3
        cumulative += liked
    return cumulative


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


# Recursive Digit Sum
#!/bin/python3


#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    x = sum(int(i) for i in n) * k
    x = str(x)
    if len(x) > 1:
        return superDigit(x, 1)
    else:
        return int(x)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3


#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#


def insertionSort1(n, arr):
    value = arr[-1]
    i = n - 2
    while (i >= 0) and (arr[i] > value):
        arr[i+1] = arr[i]
        print(*arr)
        i -= 1
    arr[i+1] = value
    print(*arr)


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2
#!/bin/python3


#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#


def insertionSort2(n, arr):
    for i in range(1, n):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        print(' '.join(map(str, arr)))


if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
