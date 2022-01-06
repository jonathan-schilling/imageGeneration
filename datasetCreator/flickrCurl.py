import urllib.request
import webbrowser
import argparse
import flickr_api
from flickr_api import Walker, Photo
import os
import json
import threading, queue

api_key = u'2cdca0964f473eb18ae03c28a5d77454'
api_secret = u'c72973370ca75da8'

extras = "tags, description, license, date_upload, date_taken, owner_name, icon_server, original_format, last_update," \
         "geo, machine_tags, o_dims, views, media, path_alias, url_l, url_c,"

sizes = {'Square': 75, 'Thumbnail': 100, 'Small': 240, 'Medium': 500, 'Medium 640': 640, 'Large': 1024, 'Original': 0}


class Continue(Exception):
    pass


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {iteration + 1}/{total + 1} {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def call_api(num_of_photos, tag_whitelist, tag_blacklist=None, output_dir="output", size="Large", create_tag_list=False,
             force_landscape=True):
    """
    call the flickr api and download num_of_photos to output_dir
    :param tag_whitelist: tags that the photos must have as csv or List
    :param tag_blacklist: tags that the photos can't have as csv or List
    :param num_of_photos: number of photos to download
    :param output_dir: path to an output dir for the photos
    :param create_tag_list: if True a file containing filename -> tags will be written in the output dir
    :param size 'Square': 75x75
                'Thumbnail': 100 on longest side
                'Small': 240 on  longest side
                'Medium': 500 on longest side
                'Medium 640': 640 on longest side
                'Large': 1024 on longest side
                'Original': original photo (not always available)
    :param force_landscape if True all photos will be in landscape with at least 16:9 aspekt ratio
    :return:
    """
    if tag_blacklist is None:
        tag_blacklist = []

    flickr_api.set_keys(api_key=api_key, api_secret=api_secret)

    """if not flickr.token_valid(perms='read'):
        # Get a request token
        flickr.get_request_token(oauth_callback='oob')

        # Open a browser at the authentication URL. Do this however
        # you want, as long as the user visits that URL.
        authorize_url = flickr.auth_url(perms='read')
        webbrowser.open_new_tab(authorize_url)

        # Get the verifier code from the user. Do this however you
        # want, as long as the user gives the application the code.
        verifier = str(input('Verifier code: '))

        # Trade the request token for an access token
        flickr.get_access_token(verifier)"""

    # print('Verification Complete')

    if isinstance(tag_whitelist, list):
        tag_whitelist = ", ".join(tag_whitelist)
    if isinstance(tag_blacklist, str):
        tag_blacklist = tag_blacklist.replace(" ", "").split(",")
    tag_blacklist = set(tag_blacklist)
    if size not in sizes.keys():
        size = "Large"

    current_photo = 0

    photo_names = set()

    photos = queue.Queue()

    issue_text = ""

    duplicate_photos = 0

    photo: Photo

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise Exception("output directory exists but is not a directory")
    else:
        os.makedirs(output_dir)

    for i, photo in enumerate(Walker(Photo.search, tag_mode='all', per_page=100, tags=tag_whitelist, extras=extras,
                                     sort="interestingness-desc")):
        printProgressBar(current_photo, num_of_photos-1, prefix="Loading Photos",
                         suffix=f"Checking Photo {i+1}, {duplicate_photos} duplicates ," + issue_text, printEnd="")

        if current_photo == num_of_photos:
            break

        try:
            photo_size = photo.getSizes()[size]
        except KeyError:
            try:
                photo_size = photo.getSizes(True)[size]
            except KeyError:
                continue

        if force_landscape and photo_size["width"] != sizes[size] or photo_size["height"] <= sizes[size] / 16 * 9:
            issue_text = "photo has the wrong size"
            continue
        try:
            if tag_blacklist:
                for tag in photo.get("tags").split(" "):
                    if tag in tag_blacklist:
                        issue_text = "photo is on the blacklist"
                        raise Continue
        except Continue:
            continue

        photo_name = f'{photo.get("id")}'

        if photo_name in photo_names:
            duplicate_photos += 1
            issue_text = "photo already written"
            continue
        photos.put(photo)
        # photo.save(os.path.join(output_dir, f"{photo_name}"), size)
        current_photo += 1
        photo_names.add(photo_name)
        issue_text = ""

    print(duplicate_photos, "duplicates found")

    def work_queue():
        while True:
            photo = photos.get()
            if create_tag_list:
                with open(os.path.join(output_dir, f"{photo_name}.json"), "w") as f:
                    f.write(json.dumps(list(map(lambda t: t.text, photo.get("tags")))))
            photo_path = os.path.join(output_dir, f'{photo.get("id")}')
            photo.save(photo_path, size)
            photos.task_done()

    for _ in range(4):
        threading.Thread(target=work_queue, daemon=True).start()

    while photos.full():
        printProgressBar(num_of_photos - photos.qsize(), num_of_photos - 1, prefix="Writing Photos", printEnd="")

    photos.join()

    print("Finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an output directory containing pictures with specific tags')
    parser.add_argument('num', type=int, help='Number of Photos to curl')
    parser.add_argument('tag', type=str, nargs='+', help='Tags that are whitelisted')
    parser.add_argument('-o', '--output', type=str, dest="output", metavar="path", default="output",
                        help="The output directory where the photos are saved. It will be created if it doesn't exist")
    parser.add_argument('-b', '--blacklist', type=str, nargs='+', dest="blacklist", metavar="tag", default=[],
                        help="The output directory where the photos are saved. It will be created if it doesn't exist")
    parser.add_argument('-bf', '--blacklist-file', type=str, dest="blacklist_file", metavar="path",
                        help="File that contains a csv blacklist")
    parser.add_argument('-t', '--tag-list', dest="tag_list", action="store_true",
                        help="A file which contains all tags for each photo will be written to the output dir")
    parser.add_argument('-l', '--landscape-only', dest="force_landscape", action="store_true",
                        help="Only output landscape pictures")
    parser.add_argument('-s', '--size', type=str, dest="size", metavar="picture size", default="Large",
                        help="""
                            'Square': 75x75
                            'Thumbnail': 100 on longest side
                            'Small': 240 on  longest side
                            'Medium': 500 on longest side
                            'Medium 640': 640 on longest side
                            'Large': 1024 on longest side
                            'Original': original photo (not always available)""")

    args = parser.parse_args()
    blacklist = args.blacklist
    if args.blacklist_file is not None:
        with open(args.blacklist_file, 'r') as f:
            try:
                blacklist += f.read().replace(" ", "").split(",")
            except Exception as e:
                print(e)
    print("The blacklist is: ", blacklist)
    call_api(args.num, args.tag, tag_blacklist=blacklist, output_dir=args.output, size=args.size,
             create_tag_list=args.tag_list, force_landscape=args.force_landscape)
