import uuid
from typing import Dict, Any, List, Union, Tuple

from spacy import Errors
from spacy.displacy.render import DEFAULT_DIR, DEFAULT_LANG
from spacy.displacy.templates import TPL_FIGURE, TPL_DEP_SVG
from spacy.tokens import Doc
from spacy.util import minify_html, escape_html
from zshot.utils.alignment_utils import filter_overlapping_spans, spacy_token_offsets
from zshot.utils.displacy.colors import light_color_from_label

from zshot.utils.displacy.templates import TPL_REL_WORDS, TPL_SCRIPT, TPL_PAGE, TPL_REL_ARCS


def parse_rels(doc: Doc) -> Dict:
    filtered_spans = filter_overlapping_spans(doc._.spans, list(doc), tokens_offsets=spacy_token_offsets(doc))
    filtered_spans.sort(key=lambda x: x.start)

    tokens_span = []
    for idx, span in enumerate(filtered_spans):
        if idx == 0:
            if span.start > 0:
                tokens_span.append((0, span.start, None))
        elif span.start > filtered_spans[idx - 1].end:
            tokens_span.append((filtered_spans[idx - 1].end, span.start, None))
        tokens_span.append((span.start, span.end, span))
    if filtered_spans[-1].end < len(doc.text):
        tokens_span.append((filtered_spans[-1].end, len(doc.text), None))

    words = []
    span_map = {}
    for idx, (start, end, span) in enumerate(tokens_span):
        words.append({
            'text': doc.text[start:end],
            'tag': span.label if span else "",
            'score': span.score if span else None,
            'color': light_color_from_label(span.label) if span else None
        }, )
        if span:
            span_map[hash(span)] = idx

    arcs = []
    for r in doc._.relations:
        idx_start = span_map[hash(r.start)]
        idx_end = span_map[hash(r.end)]
        if idx_start <= idx_end:
            arc = {'start': idx_start, 'end': idx_end,
                   'label': r.relation.name, 'dir': 'right', 'score': r.score}
        else:
            arc = {'end': idx_start, 'start': idx_end,
                   'label': r.relation.name, 'dir': 'left', 'score': r.score}
        arcs.append(arc)

    return {'words': words,
            'arcs': arcs,
            'settings': {'lang': 'en', 'direction': 'ltr'}}


class RelationsRenderer:
    """Render relations parses as SVGs."""

    style = "rel"

    def __init__(self, options: Dict[str, Any] = {}) -> None:
        """Initialise relations renderer.

        options (dict): Visualiser-specific options (compact, word_spacing,
            arrow_spacing, arrow_width, arrow_stroke, distance, offset_x,
            color, bg, font)
        """
        self.compact = options.get("compact", False)
        self.score = options.get("score", False)
        self.word_spacing = options.get("word_spacing", 30)
        self.arrow_spacing = options.get("arrow_spacing", 12 if self.compact else 20)
        self.arrow_width = options.get("arrow_width", 6 if self.compact else 10)
        self.arrow_stroke = options.get("arrow_stroke", 2)
        self.distance = options.get("distance", 150 if self.compact else 175)
        self.offset_x = options.get("offset_x", 50)
        self.color = options.get("color", "#000000")
        self.bg = options.get("bg", "#ffffff")
        self.font = options.get("font", "Arial")
        self.direction = DEFAULT_DIR
        self.lang = DEFAULT_LANG

    def render(
            self, parsed: List[Dict[str, Any]], page: bool = False, minify: bool = False
    ) -> str:
        """Render complete markup.

        parsed (list): Dependency parses to render.
        page (bool): Render parses wrapped as full HTML page.
        minify (bool): Minify HTML markup.
        RETURNS (str): Rendered SVG or HTML markup.
        """
        # Create a random ID prefix to make sure parses don't receive the
        # same ID, even if they're identical
        id_prefix = uuid.uuid4().hex
        rendered = []
        for i, p in enumerate(parsed):
            if i == 0:
                settings = p.get("settings", {})
                self.direction = settings.get("direction", DEFAULT_DIR)
                self.lang = settings.get("lang", DEFAULT_LANG)
            render_id = f"{id_prefix}-{i}"
            svg = self.render_svg(render_id, p["words"], p["arcs"])
            rendered.append(svg)
        if page:
            content = "".join([TPL_FIGURE.format(content=svg) for svg in rendered]) + TPL_SCRIPT
            markup = TPL_PAGE.format(
                content=content, lang=self.lang, dir=self.direction
            )
        else:
            markup = "".join(rendered) + TPL_SCRIPT
        if minify:
            return minify_html(markup)
        return markup

    def render_svg(
            self,
            render_id: Union[int, str],
            words: List[Dict[str, Any]],
            arcs: List[Dict[str, Any]],
    ) -> str:
        """Render SVG.

        render_id (Union[int, str]): Unique ID, typically index of document.
        words (list): Individual words and their tags.
        arcs (list): Individual arcs and their start, end, direction and label.
        RETURNS (str): Rendered SVG markup.
        """
        self.levels = self.get_levels(arcs)
        self.highest_level = max(self.levels.values(), default=0)
        self.offset_y = self.distance / 2 * self.highest_level + self.arrow_stroke
        self.width = self.offset_x + len(words) * self.distance
        self.height = self.offset_y + 3 * self.word_spacing
        self.id = render_id
        words_svg = [
            self.render_span(w["text"], w["tag"], i, w.get("score", 1), w.get("color", "#00000000"))
            for i, w in enumerate(words)
        ]
        arcs_svg = [
            self.render_arrow(a["label"], a["start"], a["end"], a["dir"], a["score"], i)
            for i, a in enumerate(arcs)
        ]
        content = "".join(words_svg) + "".join(arcs_svg)
        return TPL_DEP_SVG.format(
            id=self.id,
            width=self.width,
            height=self.height,
            color=self.color,
            bg=self.bg,
            font=self.font,
            content=content,
            dir=self.direction,
            lang=self.lang,
        )

    def render_span(self, text: str, tag: str, i: int, score: float, color: str = "#00000000") -> str:
        """
        Render a span
        :param text:
        :param tag:
        :param i:
        :param score:
        :param color:
        :return:
        """
        y = self.offset_y + self.word_spacing
        x = self.offset_x if i == 0 else self.word_spacing
        html_text = escape_html(text)
        return TPL_REL_WORDS.format(text=html_text,
                                    tag=tag + (" ({:.1f})".format(score) if score and self.score else ""), x=x, y=y,
                                    id=i, bg=color)

    def render_arrow(
            self, label: str, start: int, end: int, direction: str, score: float, i: int
    ) -> str:
        """Render individual arrow.

        label (str): Dependency label.
        start (int): Index of start word.
        end (int): Index of end word.
        direction (str): Arrow direction, 'left' or 'right'.
        i (int): Unique ID, typically arrow index.
        RETURNS (str): Rendered SVG markup.
        """
        if start < 0 or end < 0:
            error_args = dict(start=start, end=end, label=label, dir=direction)
            raise ValueError(Errors.E157.format(**error_args))
        level = self.levels[(start, end, label)]
        x_start = self.offset_x + start * self.distance + self.arrow_spacing
        y = self.offset_y
        x_end = (self.offset_x + (end - start) * self.distance + start * self.
                 distance - self.arrow_spacing * (self.highest_level - level) / 4)
        y_curve = self.offset_y - level * self.distance / 2
        if self.compact:
            y_curve = self.offset_y - level * self.distance / 6
        if y_curve == 0 and max(self.levels.values(), default=0) > 5:
            y_curve = -self.distance
        arrowhead = self.get_arrowhead(direction, x_start, y, x_end)
        arc = self.get_arc(x_start, y, y_curve, x_end)
        label_side = "right" if self.direction == "rtl" else "left"
        return TPL_REL_ARCS.format(
            id=self.id,
            i=i,
            stroke=self.arrow_stroke,
            head=arrowhead,
            label=label + (" ({:.1f})".format(score) if score and self.score else ""),
            label_side=label_side,
            arc=arc,
            start=start,
            end=end,
            direction=direction
        )

    def get_arc(self, x_start: int, y: int, y_curve: int, x_end: int) -> str:
        """Render individual arc.

        x_start (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        y_curve (int): Y-corrdinate of Cubic Bézier y_curve point.
        x_end (int): X-coordinate of arrow end point.
        RETURNS (str): Definition of the arc path ('d' attribute).
        """
        template = "M{x},{y} C{x},{c} {e},{c} {e},{y}"
        return template.format(x=x_start, y=y, c=y_curve, e=x_end)

    def get_arrowhead(self, direction: str, x: int, y: int, end: int) -> str:
        """Render individual arrow head.

        direction (str): Arrow direction, 'left' or 'right'.
        x (int): X-coordinate of arrow start point.
        y (int): Y-coordinate of arrow start and end point.
        end (int): X-coordinate of arrow end point.
        RETURNS (str): Definition of the arrow head path ('d' attribute).
        """
        if direction == "left":
            p1, p2, p3 = (x, x - self.arrow_width + 2, x + self.arrow_width - 2)
        else:
            p1, p2, p3 = (end, end + self.arrow_width - 2, end - self.arrow_width + 2)
        return f"M{p1},{y + 2} L{p2},{y - self.arrow_width} {p3},{y - self.arrow_width}"

    def get_levels(self, arcs: List[Dict[str, Any]]) -> Dict[Tuple[int, int, str], int]:
        """Calculate available arc height "levels".
        Used to calculate arrow heights dynamically and without wasting space.

        args (list): Individual arcs and their start, end, direction and label.
        RETURNS (dict): Arc levels keyed by (start, end, label).
        """
        arcs = [dict(t) for t in {tuple(sorted(arc.items())) for arc in arcs}]
        length = max([arc["end"] for arc in arcs], default=0)
        max_level = [0] * length
        levels = {}
        for arc in sorted(arcs, key=lambda arc: arc["end"] - arc["start"]):
            level = max(max_level[arc["start"]: arc["end"]]) + 1
            for i in range(arc["start"], arc["end"]):
                max_level[i] = level
            levels[(arc["start"], arc["end"], arc["label"])] = level
        return levels
