#  date: 4. 5. 2023
#  author: Daniel Schnurpfeil
#
import os

from historical_map_border_detection import main
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='historical map border detection')
    parser.add_argument('-i', '--input_directory_path',
                        required=True)
    parser.add_argument('-o', '--output_directory_path',
                        required=True)
    parser.add_argument('--min_longest_liner_means_percentile',
                        default=95, type=int)
    parser.add_argument('--min_longest_liner_extremes_percentile',
                        default=30, type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.input_directory_path):
        raise "bad input_file_path..."

    if not os.path.isdir(args.output_directory_path):
        raise "bad input_file_path..."

    img_names = os.listdir(args.input_directory_path)

    for image_name in img_names:
        input_picture_path = args.input_directory_path + image_name
        main(image_name=image_name, input_picture_path=input_picture_path, out_dir_path=args.output_directory_path,
             min_longest_liner_means_percentile=args.min_longest_liner_means_percentile,
             min_longest_liner_extremes_percentile=args.min_longest_liner_extremes_percentile
             )
