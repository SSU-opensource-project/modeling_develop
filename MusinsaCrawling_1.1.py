from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import time
import urllib.request
import sys  # 잠시 출력 텍스트값 저장을 위함

sys.stdout = open('Tmp DB.txt', 'w')


# itemName: 제품 카테고리, pageNum: 무신사에서 제공하는 페이지 값


def PageUrl(itemName, pageNum):
    url = "https://search.musinsa.com/search/musinsa/goods?q=" + itemName + "&list_kind=small&sortCode=pop&sub_sort=&page=" + \
        str(pageNum) + "&display_cnt=0&saleGoods=false&includeSoldOut=false&popular=false&category1DepthCode=&category2DepthCodes=&category3DepthCodes=&selectedFilters=&category1DepthName=&category2DepthName=&brandIds=&price=&colorCodes=&contentType=&styleTypes=&includeKeywords=&excludeKeywords=&originalYn=N&tags=&saleCampaign=false&serviceType=&eventType=&type=&season=&measure=&openFilterLayout=N&selectedOrderMeasure=&d_cat_cd="
    return url


# Chrome으로 검색하기 위함. os.getcwd()는 현재 디렉토리, /chromdriver은 크롬드라이버(다운필요)
driver = webdriver.Chrome(os.getcwd() + "/chromedriver")

FindingItemName = "상의"  # 카테고리별로 검색 가능. 근데 입력잘 해야함

pageUrl = PageUrl(FindingItemName, 1)
driver.get(pageUrl)


totalPageNum = driver.find_element_by_css_selector(".totalPagingNum").text
print("Total Page of ", FindingItemName, " : ", str(totalPageNum))

# 실험에서는 2페이지만하기

for i in range(3):  # for i in range(int(totalPageNum)):
    pageUrl = PageUrl(FindingItemName, i+1)
    driver.get(pageUrl)
    time.sleep(1)  # 이미지 로딩동안 기다리기

    item_infos = driver.find_elements_by_css_selector(
        ".img-block")  # 한 페이지에 있는 이미지 개수
    item_images = driver.find_elements_by_css_selector(
        ".lazyload.lazy")  # 이미지 가져옴

    print("Finding: ", FindingItemName, " - Page ", i+1, "/", totalPageNum)
    print(" start - ", len(item_infos), " items exist")

    for j in range(len(item_infos)):  # 페이지에 있는 이미지 개수만큼
        try:

            title = item_infos[j].get_attribute("title")  # 제품 이름
            price = item_infos[j].get_attribute("data-bh-content-meta3")  # 가격
            item_url = item_infos[j].get_attribute(
                "href")  # 무신사에서 눌렀을때 그 제품 url
            img_url = item_images[j].get_attribute(
                "data-original")  # 이미지 자체의 url (다운)

            print(title, price, item_url, img_url)
            # print("Title: ", title)
            # print("Price: ", price)
            # print("item URL: ", item_url)
            # print("Image URL: ", img_url)
            # print()
            # 이미지 다운로드. URL도 원하면 다운받을 수 있을듯
            urllib.request.urlretrieve(
                img_url, FindingItemName + str(i+1) + " Page "+str(j+1)+".jpg")

        except Exception as e:  # 에러떠도 일단 패스
            print(e)
            pass
sys.stdout.close()
driver.close()

# elem = driver.find_element_by_name("q")
# elem.send_keys("조코딩")
# elem.send_keys(Keys.RETURN)
# images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
# count = 1
# for image in images:
#     image.click()
#     time.sleep(2)
#     imgUrl = driver.find_element_by_css_selector(
#         ".n3VNCb").get_attribute("src")
#     count = count + 1
#     opener = urllib.request.build_opener()
#     opener.addheaders = [
#         ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
#     urllib.request.install_opener(opener)
#     urllib.request.urlretrieve(imgUrl, str(count)+".jpg")


# assert "Python" in driver.title
# elem = driver.find_element_by_name("q")
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
# driver.close()
