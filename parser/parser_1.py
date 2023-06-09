import time
import requests
from threading import *
import glob
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from PIL import Image
from io import BytesIO

PANORAMAS_PATH =    # path to the folder where panoramas will be stored
MAP_URL =           # geoguessr map url, for example "https://www.geoguessr.com/maps/world/play"
EMAIL =             # geoguessr account email
PASSWORD =          # geoguessr account password

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')

browser = webdriver.Chrome(chrome_options=options)
actions = ActionChains(browser)
wait = WebDriverWait(browser, 5)

def geoguessr_login(email, password):
    browser.get("https://www.geoguessr.com/signin")
    email_name = "email"
    browser.find_element(By.NAME, email_name).send_keys(email)
    pass_name = "password"
    browser.find_element(By.NAME, pass_name).send_keys(password)
    login_button_class = "button_wrapper__NkcHZ"
    browser.find_element(By.CLASS_NAME, login_button_class).click()

    time.sleep(5)

def join_map(map_url):
    browser.get(map_url)
    start_button_class_name = "button_wrapper__NkcHZ"
    browser.find_element(By.CLASS_NAME, start_button_class_name).click()

def refresh(map_url):
    """
    Working in another thread, refreshes the page every n seconds.
    Prevents random geoguessr errors.

    """
    time.sleep(300)
    join_map(map_url)
    refresh(map_url)

def interceptor(request):
    """
    Intercepts request from Google API.
    Gets coordinates and panoid using parse_coordinates_from_response and parse_panoid_from_response.
    Call image_grabber.

    """
    if 'https://maps.googleapis.com/maps/api/js/GeoPhotoService.GetMetadata' in request.url:
        response = requests.get(request.url, allow_redirects=True)
        coordinates = parse_coordinates_from_response(str(response.content))
        panoid = parse_panoid_from_response(str(response.content))
        panoimg = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={panoid}&output=tile&x=0&y=0&zoom=0&nbt=1&fover=2"
        image_grabber(panoimg, coordinates)

def parse_panoid_from_response(raw_content):
    first_relevant_byte = raw_content.find('"') + 1
    panoid = str(raw_content[first_relevant_byte:first_relevant_byte+22])

    return panoid

def parse_coordinates_from_response(raw_content):
    first_relevant_byte = raw_content.find('Google') + 30
    relevant_info = str(
        raw_content[first_relevant_byte:first_relevant_byte+400])
    coordinates = str(relevant_info[:relevant_info.find(']')])

    return coordinates

def image_grabber(panoimg, coordinates):
    """
    If there are no similar coordinates :
        Since there is a black part in the panorama, crops it and saves in folder.

    """
    if check_coords(coordinates):
        r = requests.get(panoimg)
        img = Image.open(BytesIO(r.content))
        img_cropped = img.crop((0, 0, 512, 208))
        img_cropped.save(f"{PANORAMAS_PATH}/{coordinates}.jpg", "JPEG")

def check_coords(coordinates):
    """
    Checks if there is a panorama with identical coordinates in the folder.
    Identical coordinates - coordinates with the same two digits after the point in latitude and longitude.

    """
    coordinates = coordinates.split(",")
    lat = coordinates[0].split(".")
    lon = coordinates[1].split(".")
    lat = f"{lat[0]}.{lat[1][:2]}" 
    lon = f"{lon[0]}.{lon[1][:2]}"
    if glob.glob(f'{lat}*,{lon}*', root_dir=PANORAMAS_PATH):
        return False
    return True

def cycle():
    map_class_name = "guess-map__canvas"
    while True:
        try:  # applies at the end of the map
            wait.until(EC.element_to_be_clickable(
                (By.CLASS_NAME, map_class_name))).click()
            actions.send_keys(Keys.SPACE).perform()
            time.sleep(1)
            actions.send_keys(Keys.SPACE).perform()
        except:  # just press space for next location
            actions.send_keys(Keys.SPACE).perform()

refresh_thread = Thread(target=refresh, args=(MAP_URL)) # using threading creating another thread
geoguessr_login(EMAIL, PASSWORD) 
browser.request_interceptor = interceptor 
refresh_thread.start()
join_map(MAP_URL)
cycle()