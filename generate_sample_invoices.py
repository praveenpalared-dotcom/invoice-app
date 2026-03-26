import os
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "sample_invoices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VENDORS = [
    "Acme Supplies Co.",
    "Global Tech Ltd.",
    "Metro Office Solutions",
    "Sunrise Logistics",
    "Pinnacle Services Inc.",
]

ITEMS_POOL = [
    ("Office Chairs",      4,  120.00),
    ("Laptop Stand",       2,   45.00),
    ("Printer Cartridges", 10,  18.50),
    ("Desk Lamps",         3,   35.00),
    ("Monitor Cables",     6,   12.00),
    ("USB Hubs",           5,   22.00),
    ("Notebooks (pack)",   8,    9.50),
    ("Pens (box)",         4,    6.00),
    ("Stapler",            2,   14.00),
    ("Paper Reams",       20,    5.50),
]


def random_date(anomaly: bool = False, future: bool = False) -> str:
    today = datetime.today()
    if future:
        delta = timedelta(days=random.randint(1, 15))
        return (today + delta).strftime("%Y-%m-%d")
    if anomaly:
        delta = timedelta(days=random.randint(95, 200))
        return (today - delta).strftime("%Y-%m-%d")
    delta = timedelta(days=random.randint(0, 60))
    return (today - delta).strftime("%Y-%m-%d")


def draw_invoice(
    draw: ImageDraw.ImageDraw,
    font_big,
    font_med,
    font_small,
    inv_number: str,
    vendor: str,
    date: str,
    items: list,
    total_override: float | None = None,
) -> None:
    W = 794

    draw.rectangle([0, 0, W, 80], fill="#1a1a2e")
    draw.text((30, 20), "INVOICE", font=font_big, fill="white")
    draw.text((550, 30), f"#{inv_number}", font=font_med, fill="#e0e0e0")

    draw.text((30, 100), f"From:  {vendor}", font=font_med, fill="#333")
    draw.text((30, 130), f"Date:  {date}", font=font_med, fill="#333")
    draw.text((30, 160), "Bill To: Accounts Payable Dept.", font=font_small, fill="#555")

    y = 210
    draw.rectangle([20, y, W - 20, y + 30], fill="#dee2e6")
    draw.text((30,  y + 6), "Description",  font=font_small, fill="#000")
    draw.text((350, y + 6), "Qty",          font=font_small, fill="#000")
    draw.text((430, y + 6), "Unit Price",   font=font_small, fill="#000")
    draw.text((580, y + 6), "Line Total",   font=font_small, fill="#000")

    y += 35
    grand_total = 0.0
    for desc, qty, unit_price in items:
        line_total = qty * unit_price
        grand_total += line_total
        draw.text((30,  y), desc,                 font=font_small, fill="#222")
        draw.text((350, y), str(qty),             font=font_small, fill="#222")
        draw.text((430, y), f"${unit_price:.2f}", font=font_small, fill="#222")
        draw.text((580, y), f"${line_total:.2f}", font=font_small, fill="#222")
        y += 28

    draw.line([20, y + 5, W - 20, y + 5], fill="#aaa", width=1)

    display_total = total_override if total_override is not None else grand_total
    draw.text((430, y + 15), "GRAND TOTAL:", font=font_med, fill="#000")
    draw.text((580, y + 15), f"${display_total:.2f}", font=font_med, fill="#c0392b")

    draw.text((30, y + 60), "Payment Terms: Net 30  |  Thank you for your business!",
              font=font_small, fill="#888")


def make_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def generate_invoice(index: int, anomaly_type: str | None = None) -> dict:
    inv_number = f"INV-{20240000 + index:08d}"
    vendor = random.choice(VENDORS)
    num_items = random.randint(1, 4)
    items = random.sample(ITEMS_POOL, num_items)
    true_total = sum(q * p for _, q, p in items)

    date = random_date()
    total_override = None
    note = "normal"

    if anomaly_type == "amount":
        total_override = true_total * 10
        note = "amount_spike"
    elif anomaly_type == "date_future":
        date = random_date(future=True)
        note = "future_date"
    elif anomaly_type == "date_old":
        date = random_date(anomaly=True)
        note = "old_date"
    elif anomaly_type == "math":
        total_override = true_total + random.uniform(50, 200)
        note = "math_error"

    img = Image.new("RGB", (794, 600), color="white")
    draw = ImageDraw.Draw(img)

    font_big = make_font(32)
    font_med = make_font(18)
    font_small = make_font(14)

    draw_invoice(draw, font_big, font_med, font_small,
                 inv_number, vendor, date, items, total_override)

    path = os.path.join(OUTPUT_DIR, f"{inv_number}.png")
    img.save(path)

    return {
        "file":          path,
        "inv_number":    inv_number,
        "vendor":        vendor,
        "date":          date,
        "true_total":    true_total,
        "display_total": total_override if total_override else true_total,
        "anomaly_note":  note,
    }


if __name__ == "__main__":
    random.seed(42)
    records = []

    for i in range(1, 31):
        records.append(generate_invoice(i))

    records.append(generate_invoice(31, "amount"))
    records.append(generate_invoice(32, "amount"))
    records.append(generate_invoice(33, "date_future"))
    records.append(generate_invoice(34, "date_old"))
    records.append(generate_invoice(35, "math"))

    dup = generate_invoice(5)
    dup_path = os.path.join(OUTPUT_DIR, "INV-20240005-DUP.png")
    os.rename(dup["file"], dup_path)
    dup["file"] = dup_path
    records.append(dup)

    print(f"✅  Generated {len(records)} invoice images in '{OUTPUT_DIR}/'")
    for r in records:
        flag = "" if r["anomaly_note"] == "normal" else f"  ⚠ {r['anomaly_note']}"
        print(f"   {r['file']}{flag}")
