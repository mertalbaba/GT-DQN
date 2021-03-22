#!/usr/bin/env python3

"""
Java-like input scanner.
The scanner breaks the input into tokens, and then converts them to
different types when requested using various next methods.
The following example allows to read a float from stdin:
    # Input:
    # 3 0.5
    sc = Scanner()
    x = sc.next_int()
    y = sc.next_float()
    type(x) is int # True
    type(y) is float # True
    x + y # 3.5
The following code allows to read until EOF and obtain int types:
    # Assume input is:
    # 10 20 30
    # 40 50 60
    sc = Scanner()
    sum = 0
    while sc.has_next():
        sum += sc.next_int()
    sum # 210
The default input stream is sys.stdin. However, it is possible
to read from a file or even a string:
    sc = Scanner(file='data.txt')
    # do stuff
    sc.close()
    # Or
    sc = Scanner(source='some string to use as input')
The scanner can also use string delimeters other than whitespace.
    sc = Scanner(delim=',')
By default, the scanner does a str split. If forced, a regex pattern can also
be used. As expected, the latter method is slower:
    content = '1 fish  2.5 fish red fish  blue fish
    sc = Scanner(source=content, delim='\S*fish\S*', force_regex=True)
    sc.next_int() # 1
    sc.has_next() # True
    sc.next_float() # 2.5
    sc.next() # red
    sc.next() # blue
    sc.has_next() # False
"""

import io, re, sys

class Scanner:

    def __init__(self, file=None, source=None,
            delim=None, force_regex=False):
        if file:
            file = open(file, 'r')
        elif source:
            file = io.StringIO(source)
        else:
            file = sys.stdin
        if force_regex and not delim:
            raise ValueError('delim must be specified with force_regex')
        self._file = file
        self._delim = delim
        self._force_regex = force_regex
        self._token = None
        self._tokens = None

    def has_next(self):
        '''Returns true if there's another token in the input.'''
        return self.peek() is not None

    def next(self):
        '''Returns the next token in the input as a string.'''
        return self.next_token()

    def next_line(self):
        '''Returns the remaining of the current line as a string.'''
        current = self.next_token()
        rest = self._delim.join(self._tokens)
        return current + rest

    def next_int(self):
        '''Returns the next token in the input as an int.'''
        return self.next_type(int)

    def next_float(self):
        '''Returns the next token in the input as a float.'''
        return self.next_type(float)

    def next_type(self, func):
        '''Convert the next token in the input as a given type func.'''
        return func(self.next_token())

    def next_token(self):
        '''Scans and returns the next token that matches the delimeter.'''
        next_token = self.peek()
        self._token = None
        return next_token

    def peek(self):
        '''Internal method. Creates a tokens iterator from the current line,
        and assigns the next token. When the iterator is finished, repeats
        the same process for the next line.'''
        if self._token:
            return self._token
        if not self._tokens:
            line = self._file.readline()
            if not line:
                return None
            # Use re split if forced, otherwise use str split
            if self._force_regex:
                splits = re.split(self._delim, line)
            else:
                splits = line.split(self._delim)
            self._tokens = iter(splits)
        try:
            self._token = next(self._tokens)
        except StopIteration:
            self._tokens = None
        # Recurse
        return self.peek()

    def is_stdin(self):
        '''Returns true if sys.stdin is being used as the input.'''
        return self._file == sys.stdin

    def close(self):
        '''Closes the scanner, including open files if any.'''
        if not self.is_stdin():
            self._file.close()