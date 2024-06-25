'''
Output missed words and percentage of missed
Sort by percentage
'''
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('transcript',
                        help='transcript with decodings')

    args = parser.parse_args()


