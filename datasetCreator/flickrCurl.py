import urllib.request
import webbrowser
import argparse
import flickrapi
import os
import json

api_key = u'2cdca0964f473eb18ae03c28a5d77454'
api_secret = u'c72973370ca75da8'

extras = "tags, description, license, date_upload, date_taken, owner_name, icon_server, original_format, last_update," \
         "geo, machine_tags, o_dims, views, media, path_alias, url_z, url_c,"


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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def call_api(num_of_photos, tag_whitelist, tag_blacklist=None, output_dir="output", create_tag_list=False):
    """
    call the flickr api and download num_of_photos to output_dir
    :param tag_whitelist: tags that the photos must have as csv or List
    :param tag_blacklist: tags that the photos can't have as csv or List
    :param num_of_photos: number of photos to download
    :param output_dir: path to an output dir for the photos
    :param create_tag_list: if True a file containing filename -> tags will be written in the output dir
    :return:
    """
    if tag_blacklist is None:
        tag_blacklist = []
    flickr = flickrapi.FlickrAPI(api_key, api_secret)
    if not flickr.token_valid(perms='read'):
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
        flickr.get_access_token(verifier)

    print('Verification Complete')

    if isinstance(tag_whitelist, list):
        tag_whitelist = ", ".join(tag_whitelist)
    if isinstance(tag_blacklist, str):
        tag_blacklist = tag_blacklist.replace(" ", "").split(",")

    max_photos = num_of_photos

    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise Exception("output directory exists but is not a directory")
    else:
        os.makedirs(output_dir)

    for i, photo in enumerate(flickr.walk(tag_mode='all', tags=tag_whitelist, extras=extras)):
        if i >= max_photos:
            break
        tags = photo.get("tags").split()
        for tag in tag_blacklist:
            if tag in tags:
                max_photos += 1
                continue

        printProgressBar(i, max_photos-1, prefix="Writing Photos", printEnd="")

        photo_name = f'{photo.get("id")}'
        try:
            urllib.request.urlretrieve(photo.get('url_z'), os.path.join(output_dir, f"{photo_name}.jpg"))
        except TypeError:
            print(photo.get('url_z'), os.path.join(output_dir, f"{photo_name}.jpg"))
            max_photos += 1
            continue

        if create_tag_list:
            with open(os.path.join(output_dir, f"{photo_name}.json"), "w") as f:
                f.write(json.dumps({s: photo.get(s) for s in extras.split(", ")}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an output directory containing pictures with specific tags')
    parser.add_argument('num', type=int, help='Number of Photos to curl')
    parser.add_argument('tag', type=str, nargs='+', help='Tags that are whitelisted')
    parser.add_argument('-o', '--output', type=str, dest="output", metavar="path", default="output",
                        help="The output directory where the photos are saved. It will be created if it doesn't exist")
    parser.add_argument('-b', '--blacklist', type=str, nargs='+', dest="blacklist", metavar="tag",
                        help="The output directory where the photos are saved. It will be created if it doesn't exist")
    parser.add_argument('-t', '--tag-list', dest="tag_list", action="store_true",
                        help="A file which contains all tags for each photo will be written to the output dir")

    args = parser.parse_args()
    call_api(args.num, args.tag, tag_blacklist=args.blacklist, output_dir=args.output, create_tag_list=args.tag_list)
