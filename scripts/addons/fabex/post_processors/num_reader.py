from . import nc_read as nc
import sys
import math

# a base class for hpgl parsers, and maybe others


class NumReader(nc.Parser):
    def __init__(self):
        nc.Parser.__init__(self)

    def get_number(self):
        number = ""

        # skip spaces and commas at start of number
        while self.line_index < self.line_length:
            c = self.line[self.line_index]
            if c == " " or c == ",":
                self.parse_word += c
            else:
                break
            self.line_index = self.line_index + 1

        while self.line_index < self.line_length:
            c = self.line[self.line_index]
            if (
                c == "."
                or c == "0"
                or c == "1"
                or c == "2"
                or c == "3"
                or c == "4"
                or c == "5"
                or c == "6"
                or c == "7"
                or c == "8"
                or c == "9"
                or c == "-"
            ):
                number += c
            else:
                break
            self.parse_word += c
            self.line_index = self.line_index + 1

        return number

    def add_word(self, color):
        self.add_text(self.parse_word, color)
        self.parse_word = ""

    def Parse(self, name, oname=None):
        self.files_open(name, oname)

        while self.readline():
            self.begin_ncblock()

            self.parse_word = ""
            self.line_index = 0
            self.line_length = len(self.line)

            while self.line_index < self.line_length:
                c = self.line[self.line_index]
                self.parse_word += c

                self.ParseFromFirstLetter(c)

                self.line_index = self.line_index + 1

            self.add_text(self.parse_word, None)

            self.end_ncblock()

        self.files_close()
