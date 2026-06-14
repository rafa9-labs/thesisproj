"""
Phase 2: Convert static var(--color-*) / var(--font-*) inline styles to Tailwind v4 utilities.

Operates on all .tsx / .ts files in frontend/src/.
Handles: color, backgroundColor, borderColor, fontFamily, background.
Leaves dynamic values (ternaries, expressions, template literals) as inline styles.
"""

import re
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "frontend" / "src"

# â”€â”€ CSS property â†’ Tailwind class prefix â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# None = special handler (fontFamily)
PROP_MAP = {
    "color": "text-({var})",
    "backgroundColor": "bg-({var})",
    "background": "bg-({var})",
    "borderColor": "border-({var})",
}

FONT_MAP = {
    "var(--font-sans)": "font-sans",
    "var(--font-mono)": "font-mono",
}

# Props we know how to convert (for safety â€” only touch these)
CONVERTIBLE = set(PROP_MAP.keys()) | {"fontFamily"}

# â”€â”€ Stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
stats = {
    "files_visited": 0,
    "files_modified": 0,
    "props_moved": 0,
    "styles_removed": 0,
    "errors": 0,
}


def find_style_blocks(text: str):
    """Yield (start, end) for every `style={{...}}` block using brace matching,
    accounting for nested braces inside strings, template literals, etc."""
    pattern = re.compile(r"style\s*=\s*(\{)")
    for m in pattern.finditer(text):
        eq_start = m.start()
        # The first { is the JSX expression brace. The second { is the object.
        # But they might be separated by whitespace.
        # Actually `style={{` â€” the first { is JSX expression start.
        # We just need to find matching }} pair.
        pos = m.end() - 1  # position of the first {
        depth = 1
        i = pos + 1
        in_string = False
        string_char = None
        in_template = False
        while i < len(text) and depth > 0:
            ch = text[i]
            if in_string:
                if ch == "\\":
                    i += 2
                    continue
                if ch == string_char:
                    in_string = False
            elif in_template:
                if ch == "${":
                    # template interpolation â€” temporarily increase brace depth
                    # skip past the interpolation
                    td = 1
                    j = i + 2
                    while j < len(text) and td > 0:
                        if text[j] == "{":
                            td += 1
                        elif text[j] == "}":
                            td -= 1
                        j += 1
                    i = j
                    continue
                if ch == "`":
                    in_template = False
            else:
                if ch == '"' or ch == "'":
                    in_string = True
                    string_char = ch
                elif ch == "`":
                    in_template = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
            i += 1
        if depth == 0:
            yield eq_start, i


def parse_style_obj(text: str):
    """Parse the content between `{{` and `}}` into a list of (key, raw_value, is_convertible).
    Returns list of dicts: {key, raw_value, is_string_static, css_var, is_font}
    """
    props = []
    # Find all key: value pairs inside the object literal
    # Match: identifierOrString: value,
    pair_pattern = re.compile(
        r'''
        ([\w-]+|"[^"]*"|'[^']*')   # key (identifier or string key)
        \s*:\s*                      # colon
        (                            # value start
            "[^"]*"                   # double-quoted string
            |'[^']*'                  # single-quoted string
            |`[^`]*`                  # template literal
            |(?:[\w$][\w$.]*\s*\()    # function call
            |\d+(?:\.\d+)?           # number
            |true|false|null          # literal
            |[\w$][\w$.<>[\]()]*      # identifier or expression
        )                            # value end
        ''',
        re.VERBOSE,
    )

    # More robust: parse char by char for key:value pairs
    i = 0
    while i < len(text):
        # Skip whitespace/comments
        while i < len(text) and text[i] in " \t\n\r,":
            i += 1
        if i >= len(text):
            break

        # Read key
        key_start = i
        if text[i] in "\"'":
            delim = text[i]
            i += 1
            while i < len(text) and text[i] != delim:
                if text[i] == "\\":
                    i += 2
                else:
                    i += 1
            if i < len(text):
                i += 1  # closing quote
        else:
            while i < len(text) and (text[i].isalnum() or text[i] in "_-"):
                i += 1
        key = text[key_start:i].strip().strip("\"'")
        if not key:
            i += 1
            continue

        # Skip colon
        while i < len(text) and text[i] in " \t\n\r":
            i += 1
        if i < len(text) and text[i] == ":":
            i += 1
        else:
            continue

        # Skip whitespace after colon
        while i < len(text) and text[i] in " \t\n\r":
            i += 1

        # Read value
        val_start = i
        if i < len(text) and text[i] == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == "\\":
                    i += 2
                else:
                    i += 1
            if i < len(text):
                i += 1
        elif i < len(text) and text[i] == "'":
            i += 1
            while i < len(text) and text[i] != "'":
                if text[i] == "\\":
                    i += 2
                else:
                    i += 1
            if i < len(text):
                i += 1
        elif i < len(text) and text[i] == "`":
            # Template literal â€” skip to end, handle interpolation
            i += 1
            td = 0
            while i < len(text):
                if text[i] == "`" and td == 0:
                    i += 1
                    break
                elif text[i] == "$" and i + 1 < len(text) and text[i + 1] == "{":
                    td += 1
                    i += 2
                elif text[i] == "{":
                    td += 1
                    i += 1
                elif text[i] == "}":
                    td -= 1
                    i += 1
                else:
                    i += 1
        elif i < len(text) and text[i] == "{":
            # Nested object
            d = 1
            i += 1
            while i < len(text) and d > 0:
                if text[i] == "{":
                    d += 1
                elif text[i] == "}":
                    d -= 1
                i += 1
        else:
            # Expression â€” read until comma or closing brace
            d = 0
            while i < len(text):
                ch = text[i]
                if ch == "," and d == 0:
                    break
                if ch == "}" and d == 0:
                    break
                if ch == "(" or ch == "[" or ch == "{":
                    d += 1
                elif ch == ")" or ch == "]" or ch == "}":
                    d -= 1
                i += 1

        raw_val = text[val_start:i].strip()

        # Classify
        is_string_static = False
        css_var = None
        is_font = False

        stripped = raw_val
        if (stripped.startswith('"') and stripped.endswith('"')) or \
           (stripped.startswith("'") and stripped.endswith("'")):
            inner = stripped[1:-1]
            # Check if it's a var(--color-*) reference
            vm = re.match(r"^var\((--color-[\w-]+)\)$", inner)
            if vm:
                is_string_static = True
                css_var = vm.group(1)
            elif inner.startswith("var(--font-"):
                is_string_static = True
                is_font = True
                css_var = inner

        props.append({
            "key": key,
            "raw": raw_val,
            "is_string_static": is_string_static,
            "css_var": css_var,
            "is_font": is_font,
        })

    return props


def convert_file(filepath: Path) -> bool:
    """Returns True if file was modified."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ERROR reading {filepath.name}: {e}")
        stats["errors"] += 1
        return False

    original = text
    modified = False

    # Process in reverse order so positions don't shift
    blocks = list(find_style_blocks(text))
    blocks.reverse()

    for eq_start, block_end in blocks:
        # Extract the style object content (between the two braces)
        # text at eq_start is like `style={{...}}`
        style_content = text[eq_start:block_end]
        # content = everything between `{{` and `}}`
        m = re.match(r"style\s*=\s*\{\{(.*)\}\}", style_content, re.DOTALL)
        if not m:
            continue
        inner = m.group(1)

        props = parse_style_obj(inner)
        convertible = [p for p in props if p["is_string_static"] and p["key"] in CONVERTIBLE]

        if not convertible:
            continue  # nothing to convert

        # Build the Tailwind classes to add
        tailwind_classes = []
        for p in convertible:
            if p["is_font"]:
                tw = FONT_MAP.get(p["css_var"])
                if tw:
                    tailwind_classes.append(tw)
                    stats["props_moved"] += 1
            elif p["key"] in PROP_MAP:
                tw = PROP_MAP[p["key"]].format(var=p["css_var"])
                tailwind_classes.append(tw)
                stats["props_moved"] += 1

        if not tailwind_classes:
            continue

        # Determine what stays in the style object
        remaining = [p for p in props if p not in convertible]

        # Build the element prefix to find className
        # Find the element opening tag before this style attribute
        pre_text = text[:eq_start]
        # Find the last < before style
        tag_start = pre_text.rfind("<")
        if tag_start == -1:
            continue

        tag_section = pre_text[tag_start:]

        # Check if there's an existing className
        has_class = bool(re.search(r'\bclassName\s*=', tag_section))

        # Build new style string
        if remaining:
            new_inner = ", ".join(f"{p['key']}: {p['raw']}" for p in remaining)
            new_style = f"style={{{{{new_inner}}}}}"
        else:
            new_style = ""  # remove style entirely

        # Find the exact span to replace: from eq_start to block_end
        old_span = text[eq_start:block_end]

        # Insert tailwind classes into className or add className
        tw_string = " ".join(tailwind_classes)

        if has_class:
            # Find className="..." or className={'...'} in the tag
            cm = re.search(r'\bclassName\s*=\s*("[^"]*"|\'[^\']*\'|\{[^}]+\})', tag_section)
            if cm:
                class_val = cm.group(1)
                if class_val.startswith("{") and class_val.endswith("}"):
                    # className={expr} â€” skip, too complex
                    # Instead, we add a separate className or skip
                    # For now, skip dynamic className
                    continue
                else:
                    # className="string"
                    inner_text = class_val[1:-1]
                    # Check if these classes are already present
                    existing = set(inner_text.split())
                    new_tw = [c for c in tailwind_classes if c not in existing]
                    if not new_tw:
                        # All already present, just remove from style
                        pass
                    new_inner = inner_text + " " + " ".join(new_tw)
                    new_inner = new_inner.strip()
                    new_class_attr = f'className="{new_inner}"'
                    # Need to replace the class attribute in the tag AND the style attribute
                    # This is getting complex with positional tracking
                    # Strategy: do the style replacement text[eq_start:block_end] = new_style
                    # and separately replace className in tag_section

                    # Let me track the className position
                    cm_start = tag_start + cm.start()
                    cm_end = tag_start + cm.end()
                    old_class_span = original[cm_start:cm_end]
                    new_class_span = new_class_attr

                    if new_style:
                        text = text[:eq_start] + new_style + text[block_end:]
                        # Adjust positions
                        diff = len(new_style) - (block_end - eq_start)
                        # Update class position
                        text = text[:cm_start] + new_class_span + text[cm_start + len(old_class_span):]
                    else:
                        # Remove style entirely
                        # Remove the style attribute including surrounding whitespace
                        # Find what to remove: from before `style` to after `}}`
                        # Build the replacement
                        # Check if there's a space before style that we should remove
                        pre_chars = text[eq_start - 1:eq_start]
                        if pre_chars in (" ", "\t"):
                            eq_start_adj = eq_start - 1
                            while eq_start_adj > 0 and text[eq_start_adj - 1] in (" ", "\t"):
                                eq_start_adj -= 1
                            text = text[:eq_start_adj] + text[block_end:]
                        else:
                            text = text[:eq_start] + text[block_end:]

                        # Update className
                        text = text[:cm_start] + new_class_span + text[cm_start + len(old_class_span):]
        else:
            # No className â€” add one
            # Insert className before style (or after tag name)
            # Find where to insert: after tag name but before any attributes
            # Simplest: insert className right before style's position
            if new_style:
                text = text[:eq_start] + f'className="{tw_string}" ' + new_style + text[block_end:]
            else:
                text = text[:eq_start] + f'className="{tw_string}"' + text[block_end:]

        modified = True

    if modified:
        filepath.write_text(text, encoding="utf-8")
        stats["files_modified"] += 1
        print(f"  âœ“ {filepath.relative_to(SRC.parent.parent)}")
        return True

    return False


def main():
    files = sorted(SRC.rglob("*.tsx")) + sorted(SRC.rglob("*.ts"))
    # Exclude node_modules, dist, etc.
    files = [f for f in files if "node_modules" not in str(f) and "dist" not in str(f)]

    print(f"Scanning {len(files)} files in {SRC}...\n")

    for fpath in files:
        stats["files_visited"] += 1
        try:
            convert_file(fpath)
        except Exception as e:
            print(f"  âœ-- ERROR on {fpath.name}: {e}")
            stats["errors"] += 1

    print("\nâ”€â”€ Conversion complete â”€â”€")
    print(f"  Files visited:  {stats['files_visited']}")
    print(f"  Files modified: {stats['files_modified']}")
    print(f"  Props moved:    {stats['props_moved']}")
    print(f"  Styles removed: {stats['styles_removed']}")
    print(f"  Errors:         {stats['errors']}")


if __name__ == "__main__":
    main()
