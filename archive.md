---
layout: page
title: Archive
---
{% assign tags=site.tags %}

<h3> Tags </h3>
{% for tag in tags %}{% capture taglink %} /tag/{{ tag[0] }}{% endcapture %} <span class='post-tag'> <a href="{{ taglink }}"> {{ tag[0] }} </a> </span> 
{% endfor %} 


<h3> Articles </h3>
{% for post in site.posts %} <span class="archivemono">{{ post.date | date_to_string }}</span> &raquo; [ {{ post.title }} ]({{ post.url }})  
{% endfor %}
