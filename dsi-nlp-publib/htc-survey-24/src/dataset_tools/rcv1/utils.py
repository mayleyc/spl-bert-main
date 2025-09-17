from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from xml.dom.minidom import Element


@dataclass
class ReuterArticle:
    """Simple class to encapsulate Reuters article data"""
    date: datetime.date
    itemid: int
    lang: str

    headline: str
    article: str

    countries: List[str]
    topics: List[str]


def parse_newsitem(dom_file) -> Optional[ReuterArticle]:
    """
    Parse a "newistem" DOM file extracted from an XML file.
    Pretty much specific to RCV2-style files, and not great, but is only run once.

    Parameters
    ----------
    dom_file : a DOM file, as parsed by minidom (belongs to python standard library)

    Returns
    -------
    obj: a ReuterArticle class instance
    """

    # Grab sub-object "newsitem". One article per xml, hence index 0
    # equivalent: dom.childNodes[0]
    # *** Generic info ***
    newsitem: Element = dom_file.getElementsByTagName("newsitem")[0]
    date: datetime.date = datetime.strptime(newsitem.getAttribute("date"), "%Y-%m-%d").date()
    itemid: int = int(newsitem.getAttribute("itemid"))
    lang: str = newsitem.getAttribute("xml:lang")

    # *** Headline ***
    headlineNode: Element = newsitem.getElementsByTagName("headline")[0]
    if not headlineNode or not headlineNode.firstChild:
        headline: str = ""
    else:
        headline: str = headlineNode.firstChild.data
    assert headlineNode.firstChild == headlineNode.lastChild, "Error: Headline node has multiple children"

    # *** Text of the article ***
    textNode: Element = newsitem.getElementsByTagName("text")[0]
    if not textNode or not textNode.firstChild:
        return None
    # The "-1" is to skip the copyright line.
    # Text nodes alternate with empty backspace nodes, hence the conditional list comprehension
    article = " ".join([x.firstChild.data.strip() for x in
                        [y for y in textNode.childNodes if isinstance(y, Element)]][:-1])

    # ** Categories & Co **
    codes: List[Element] = [c for c in newsitem.getElementsByTagName("metadata")[0].getElementsByTagName("codes")]
    # An ugly assert that breaks for EN ( ੭눈 _ 눈 )੭
    # assert [c.getAttribute('class') for c in codes] == ['bip:countries:1.0', 'bip:topics:1.0'], ("Found an "
    #                                                                                            "uncomforing XML file")
    # EN files also have a 'bip:industries:1.0' attribute, RCV2 fused them with topics. They are discarded.
    code_labels = [c.getAttribute("class") for c in codes]
    # Extract topics and instances
    # No topics lead to the article being discarded in the next step
    if "bip:topics:1.0" not in code_labels:
        return None
    else:
        topicNodes = codes[code_labels.index("bip:topics:1.0")]
        topics: List[str] = [c.getAttribute("code") for c in topicNodes.getElementsByTagName("code")]
        # No topics is fine, but needs to be handled
        if "bip:countries:1.0" in code_labels:
            countryNodes = codes[code_labels.index("bip:countries:1.0")]
            countries: List[str] = [c.getAttribute("code") for c in countryNodes.getElementsByTagName("code")]
        else:
            countries = []

    # Create data class instance
    obj = ReuterArticle(date, itemid, lang, headline, article, countries, topics)
    return obj
