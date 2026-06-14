"""
Phase 3: Safe style elimination. Converts style={{}} blocks ONLY when ALL
properties are static and have Tailwind equivalents. Multi-line aware.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"

RULES = [
    # (property, regex_for_value, tailwind_class_fn)
    ("fontSize",      r"^(\d+)$",                   lambda v: f"text-[{v}px]"),
    ("fontWeight",    r"^(400|500|600|700|800)$",    lambda v: {"400":"font-normal","500":"font-medium","600":"font-semibold","700":"font-bold","800":"font-extrabold"}[v]),
    ("letterSpacing", r'^(-?\d+\.?\d*em)$',       lambda v: f"tracking-[{v}]"),
    ("textTransform", r"^uppercase$",              lambda _: "uppercase"),
    ("textTransform", r"^capitalize$",             lambda _: "capitalize"),
    ("textAlign",     r"^right$",                  lambda _: "text-right"),
    ("textAlign",     r"^center$",                 lambda _: "text-center"),
    ("textAlign",     r"^left$",                   lambda _: "text-left"),
    ("display",       r"^flex$",                   lambda _: "flex"),
    ("display",       r"^block$",                  lambda _: "block"),
    ("display",       r"^none$",                   lambda _: "hidden"),
    ("display",       r"^inline-flex$",            lambda _: "inline-flex"),
    ("flexShrink",    r"^0$",                      lambda _: "shrink-0"),
    ("flex",          r"^1$",                      lambda _: "flex-1"),
    ("flexDirection", r"^column$",                 lambda _: "flex-col"),
    ("flexDirection", r"^row$",                    lambda _: "flex-row"),
    ("alignItems",    r"^center$",                 lambda _: "items-center"),
    ("alignItems",    r"^flex-start$",             lambda _: "items-start"),
    ("alignItems",    r"^flex-end$",               lambda _: "items-end"),
    ("justifyContent",r"^center$",                 lambda _: "justify-center"),
    ("justifyContent",r"^space-between$",          lambda _: "justify-between"),
    ("justifyContent",r"^flex-end$",               lambda _: "justify-end"),
    ("opacity",       r"^(0\.\d+)$",               lambda v: f"opacity-{int(float(v)*100)}"),
    ("cursor",        r"^pointer$",                lambda _: "cursor-pointer"),
    ("cursor",        r"^not-allowed$",            lambda _: "cursor-not-allowed"),
    ("cursor",        r"^default$",                lambda _: "cursor-default"),
    ("overflow",      r"^hidden$",                 lambda _: "overflow-hidden"),
    ("overflow",      r"^auto$",                   lambda _: "overflow-auto"),
    ("overflowX",     r"^hidden$",                 lambda _: "overflow-x-hidden"),
    ("overflowY",     r"^auto$",                   lambda _: "overflow-y-auto"),
    ("position",      r"^relative$",               lambda _: "relative"),
    ("position",      r"^absolute$",               lambda _: "absolute"),
    ("whiteSpace",    r"^nowrap$",                 lambda _: "whitespace-nowrap"),
    ("outline",       r"^none$",                   lambda _: "outline-none"),
    ("borderCollapse",r"^collapse$",               lambda _: "border-collapse"),
    ("textOverflow",  r"^ellipsis$",               lambda _: "text-ellipsis"),
    ("width",         r"^(\d+)$",                  lambda v: f"w-[{v}px]"),
    ("width",         r"^100%$",                   lambda _: "w-full"),
    ("height",        r"^(\d+)$",                  lambda v: f"h-[{v}px]"),
    ("height",        r"^100%$",                   lambda _: "h-full"),
    ("minWidth",      r"^(\d+)$",                  lambda v: f"min-w-[{v}px]"),
    ("maxHeight",     r"^(\d+)$",                  lambda v: f"max-h-[{v}px]"),
    ("borderRadius",  r"^(\d+)$",                  lambda v: f"rounded-[{v}px]"),
    ("zIndex",        r"^(\d+)$",                  lambda v: f"z-[{v}]"),
    ("gap",           r"^(\d+)$",                  lambda v: f"gap-[{v}px]"),
    ("margin",        r"^(\d+)$",                  lambda v: f"m-[{v}px]"),
    ("marginTop",     r"^(\d+)$",                  lambda v: f"mt-[{v}px]"),
    ("marginBottom",  r"^(\d+)$",                  lambda v: f"mb-[{v}px]"),
    ("marginLeft",    r"^(\d+)$",                  lambda v: f"ml-[{v}px]"),
    ("marginRight",   r"^(\d+)$",                  lambda v: f"mr-[{v}px]"),
    ("padding",       r"^(\d+)$",                  lambda v: f"p-[{v}px]"),
    ("paddingTop",    r"^(\d+)$",                  lambda v: f"pt-[{v}px]"),
    ("paddingBottom", r"^(\d+)$",                  lambda v: f"pb-[{v}px]"),
    ("paddingLeft",   r"^(\d+)$",                  lambda v: f"pl-[{v}px]"),
    ("paddingRight",  r"^(\d+)$",                  lambda v: f"pr-[{v}px]"),
    ("boxShadow",     r'^"([^"]+)"$',              lambda v: f"shadow-[{v.replace(' ','_')}]"),
    ("background",    r"^transparent$",            lambda _: "bg-transparent"),
    ("background",    r"^none$",                   lambda _: "bg-none"),
    ("lineHeight",    r"^([\d.]+)$",               lambda v: f"leading-[{v}]"),
    ("colorScheme",   r"^dark$",                   lambda _: "[color-scheme:dark]"),
    ("accentColor",   r"^var\(--color-brand\)$",   lambda v: f"accent-({v})"),
]

stats = {"files": 0, "blocks": 0, "props": 0, "errors": 0}


def parse_props(inner):
    pairs = []
    current = ""
    depth = 0
    in_str = False
    sq = None
    for ch in inner:
        if in_str:
            current += ch
            if ch == "\\":
                continue
            if ch == sq:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                sq = ch
                current += ch
            elif ch in "({[":
                depth += 1
                current += ch
            elif ch in ")}]":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                part = current.strip()
                if part:
                    pairs.append(part)
                current = ""
            elif ch == "\n":
                current += " "
            else:
                current += ch
    if current.strip():
        pairs.append(current.strip())

    result = []
    for part in pairs:
        idx = part.find(":")
        if idx == -1:
            continue
        key = part[:idx].strip()
        val = part[idx + 1:].strip()
        result.append((key, val))
    return result


def is_static(val):
    v = val.strip()
    if v.startswith('"') and v.endswith('"'):
        return True
    if re.match(r"^-?\d+(\.\d+)?$", v):
        return True
    # Unquoted identifiers
    if re.match(r"^[a-zA-Z][\w-]*$", v):
        return True
    return False


def all_convertible(props):
    tw_classes = []
    for key, val in props:
        if not is_static(val):
            return None
        clean_val = val.strip().strip('"').strip("'")
        matched = False
        for pname, regex, fn in RULES:
            if key != pname:
                continue
            m = re.match(regex, clean_val)
            if m:
                groups = m.groups()
                tw = fn(groups[0] if groups else "")
                if tw:
                    tw_classes.append(tw)
                matched = True
                break
        if not matched:
            return None
    return tw_classes


def process_file(filepath):
    try:
        text = filepath.read_text(encoding="utf-8")
    except:
        stats["errors"] += 1
        return

    original = text
    replacements = []

    # Find style={{...}} blocks
    i = 0
    while i < len(text):
        idx = text.find("style={{", i)
        if idx == -1:
            break
        brace_start = idx + 8
        depth = 1
        j = brace_start
        in_str = False
        sq = None
        while j < len(text) and depth > 0:
            ch = text[j]
            if in_str:
                if ch == "\\":
                    j += 2
                    continue
                if ch == sq:
                    in_str = False
            else:
                if ch in ('"', "'", "`"):
                    in_str = True
                    sq = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            j += 1

        if depth != 0:
            i = j
            continue

        inner = text[brace_start : j - 2]
        i = j

        props = parse_props(inner)
        if not props:
            continue

        tw_classes = all_convertible(props)
        if not tw_classes:
            continue

        cls_str = " ".join(tw_classes)
        stats["blocks"] += 1
        stats["props"] += len(tw_classes)

        # Find className in the parent tag
        pre = text[:idx]
        last_open = pre.rfind("<")
        if last_open == -1:
            continue

        tag_region = text[last_open:idx]
        cn_match = re.search(r'className\s*=\s*"([^"]*)"', tag_region)

        # Remove whitespace before style block
        rs = idx
        while rs > 0 and text[rs - 1] in (" \t"):
            rs -= 1

        if cn_match:
            existing = cn_match.group(1).strip()
            new_cn = existing
            for c in tw_classes:
                if c not in new_cn:
                    new_cn = new_cn + " " + c if new_cn else c
            new_cn = new_cn.strip()
            cn_start = last_open + cn_match.start()
            cn_end = last_open + cn_match.end()
            # Two non-overlapping replacements
            replacements.append((cn_start, cn_end, f'className="{new_cn}"'))
            # Remove style (cn_end < rs guaranteed since className is before style)
            replacements.append((rs, j, ""))
        else:
            # Insert className where style was
            nl = text.rfind("\n", 0, idx)
            indent = " "
            if nl != -1:
                ws = text[nl + 1 : idx]
                stripped = ws.lstrip()
                indent = ws[: len(ws) - len(stripped)]
            replacements.append((rs, j, f'{indent}className="{cls_str}" '))

    if not replacements:
        return

    # Sort descending and apply — no merging needed (replacements don't overlap)
    replacements.sort(key=lambda x: x[0], reverse=True)
    for s, e, r in replacements:
        text = text[:s] + r + text[e:]

    if text != original:
        filepath.write_text(text, encoding="utf-8")
        stats["files"] += 1


def main():
    files = sorted(SRC.rglob("*.tsx"))
    files = [f for f in files if "node_modules" not in str(f) and "dist" not in str(f)]

    print(f"Safe elimination across {len(files)} files...\n")

    for fpath in files:
        try:
            process_file(fpath)
        except Exception as e:
            print(f"  ERR {fpath.name}: {e}")
            stats["errors"] += 1

    print(f"\n-- Done --")
    print(f"  Files modified:   {stats['files']}")
    print(f"  Style blocks converted: {stats['blocks']}")
    print(f"  Props converted:  {stats['props']}")
    print(f"  Errors:           {stats['errors']}")


if __name__ == "__main__":
    main()
