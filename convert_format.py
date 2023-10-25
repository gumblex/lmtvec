import sys
import logging
import argparse
from lmtvec import textfmt, ftbin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert word vector to lmtvec format.")
    parser.add_argument('-f', '--format', choices=('text', 'fasttext'),
                        default='text', help='Input format')
    parser.add_argument('-c', '--chunk', type=int,
                        default=25000, help='Chunk size')
    parser.add_argument('-i', '--input', default='-',
                        help='Input file, use - for stdin')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr, format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )

    if args.input == '-':
        input_file = sys.stdin.buffer
    else:
        input_file = open(args.input, 'rb')
    if args.format == 'text':
        textfmt.convert_from_text(args.output, input_file, args.chunk)
    else:
        ftbin.convert_from_fasttext_binary(args.output, input_file, args.chunk)

