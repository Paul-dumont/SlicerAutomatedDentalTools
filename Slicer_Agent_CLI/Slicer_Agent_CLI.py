#!/usr/bin/env python-real

import argparse

def main(args)-> None:

    msg =  args.msg
    print("CLI :" + msg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('msg',type=str, help='input message')


    args = parser.parse_args()
    main(args)