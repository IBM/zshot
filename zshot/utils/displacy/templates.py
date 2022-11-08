TPL_PAGE = """
<!DOCTYPE html>
<html lang="{lang}">
    <head>
        <title>Zshot displaCy</title>
    </head>

    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: {dir}">{content}</body>
</html>
"""

TPL_REL_WORDS = """
<text id="{id}" class="displacy-token" fill="currentColor" text-anchor="middle" y="{y}" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    <tspan class="displacy-word" fill="currentColor" x="{x}">{text}</tspan>
</text>
<text id="tag-{id}" class="displacy-tag" fill="currentColor" text-anchor="middle" y="{y}">
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="{x}">{tag}</tspan>
</text>
"""

TPL_REL_ARCS = """
<g class="displacy-arrow" start={start} end={end} direction={direction}>
    <path class="displacy-arc" id="arrow-{id}-{i}" stroke-width="{stroke}px" d="{arc}" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-{id}-{i}" class="displacy-label" startOffset="50%" side="{label_side}" fill="currentColor" text-anchor="middle">{label}</textPath>
    </text>
    <path class="displacy-arrowhead" d="{head}" fill="currentColor"/>
</g>
"""

TPL_SCRIPT = """
<script>
    function positionSpan(elem) {
        var bounds = elem.getBBox();
        var x = bounds.x + bounds.width/2;
        var id = parseInt(elem.getAttribute("id"));
        if (elem.classList.contains('displacy-token')){
            if(id == 0) {
                elem.firstElementChild.setAttribute("x", parseInt(bounds.x + bounds.width));
            }else{
                var prevSpan = document.getElementById(`${id-1}`);
                var prevX = prevSpan.firstElementChild.getAttribute("x")
                prevBounds = prevSpan.getBBox();
                elem.firstElementChild.setAttribute("x", parseFloat(elem.firstElementChild.getAttribute("x")) + parseFloat(prevX) + parseFloat(prevBounds.width/2) + parseFloat(bounds.width/2));
            }
            var tagElem = document.getElementById(`tag-${id}`);
            tagElem.firstElementChild.setAttribute("x", parseInt(elem.firstElementChild.getAttribute("x")));

            if(document.getElementById(`${id+1}`) == undefined){
                var svg = document.querySelectorAll("svg");
                var initSpan = document.getElementById(0);
                var padding = parseInt(initSpan.firstElementChild.getAttribute("x")) - parseInt(initSpan.getBBox().width/2);
                svg[0].setAttribute("width", `${parseFloat(elem.firstElementChild.getAttribute("x")) + parseFloat(bounds.width/2) + padding}`);
            }
        }
    }
    
    function makeBG(elem) {
        var svgns = "http://www.w3.org/2000/svg"
        var bounds = elem.getBBox()
        var bg = document.createElementNS(svgns, "rect")
        var style = getComputedStyle(elem)
        var padding_top = parseInt(style["padding-top"])
        var padding_left = parseInt(style["padding-left"])
        var padding_right = parseInt(style["padding-right"])
        var padding_bottom = parseInt(style["padding-bottom"])
        bg.setAttribute("x", bounds.x - parseInt(style["padding-left"]))
        bg.setAttribute("y", bounds.y - parseInt(style["padding-top"]))
        bg.setAttribute("width", bounds.width + padding_left + padding_right)
        bg.setAttribute("height", bounds.height + padding_top + padding_bottom)
        bg.setAttribute("fill", style["background-color"])
        bg.setAttribute("rx", style["border-radius"])
        bg.setAttribute("stroke-width", style["border-top-width"])
        bg.setAttribute("stroke", style["border-top-color"])
        if (elem.hasAttribute("transform")) {
            bg.setAttribute("transform", elem.getAttribute("transform"))
        }
        elem.parentNode.insertBefore(bg, elem)
    }
    
    function positionArc(elem) {
        if (elem != undefined && elem.classList.contains('displacy-arrow')){
            startId = elem.getAttribute("start");
            endId = elem.getAttribute("end");
            direction = elem.getAttribute("direction");
            var startSpan = document.getElementById(startId);
            var endSpan = document.getElementById(endId);
            var xStart = parseFloat(startSpan.firstElementChild.getAttribute("x"));
            var xEnd = parseFloat(endSpan.firstElementChild.getAttribute("x"));
            var arc = elem.firstElementChild.getAttribute("d");
            
            var map = {start:xStart, end:xEnd};
            arc = arc.replace(/[M](\w+),/g,"M{start},").replace(/[C](\w+),/g,"C{start},").replace(/ (\d+\.?\d+)/g," {end}");
            arc = arc.replace(/{(\w+)}/g, function(_,k){
              return map[k];
            });
            elem.firstElementChild.setAttribute("d", arc);
            
            // Move Head
            var headElem = elem.querySelectorAll(".displacy-arrowhead")[0];
            var head = headElem.getAttribute("d");
            head = head.replace(/M(\d+\.?\d+),/g,"M{p1},").replace(/[L](\d+\.?\d+),/g,"L{p2},").replace(/ (\d+\.?\d+)/g," {p3}");
            var arrow_width = parseFloat(headElem.getBBox().width);
            var map = {p1:xEnd, p2:(xEnd + arrow_width - 2), p3:(xEnd - arrow_width + 2)};
            if (direction == "left"){
                map = {p1:xStart, p2:(xStart + arrow_width - 2), p3:(xStart - arrow_width + 2)};
            }
            head = head.replace(/{(\w+)}/g, function(_,k){
              return map[k];
            });
            headElem.setAttribute("d", head);
        }
    }

    var texts = document.querySelectorAll("text");
    
    for (var i = 0; i < texts.length; i++) {
        positionSpan(texts[i])
    }
    
    for (var i = 0; i < texts.length; i++) {
        makeBG(texts[i])
    }
    
    var arcs = document.querySelectorAll("g");
    for (var i = 0; i < texts.length; i++) {
        positionArc(arcs[i])
    }
</script>
"""  # noqa
