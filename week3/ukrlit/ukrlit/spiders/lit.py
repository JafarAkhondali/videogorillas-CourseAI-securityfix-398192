# -*- coding: utf-8 -*-
import scrapy
import re


class LitSpider(scrapy.Spider):
    name = 'lit'
    allowed_domains = ['ukrlit.org']
    start_urls = ['http://ukrlit.org/']

    def parse(self, response):
        if response.url.endswith('.txt'):
            fn = 'texts/' + response.url.split('/')[-1]
            html = response.body.decode('utf-8')
            html = html.replace('УКРЛІТ.ORG — публічна електронна бібліотека української художньої літератури.', '')
            html = re.sub(r'Постійна адреса: http://ukrlit.org/.*', '', html)
            html = html.strip()
            if len(html):
                with open(fn, 'w') as f:
                    f.write(html)

        for link in response.css('a::attr(href)').extract():
            if not re.search(r'/\d+$', link) and not link.endswith('.pdf') and not 'slovnyk' in link:
                yield response.follow(link)
