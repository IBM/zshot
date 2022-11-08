import zlib


def light_color_from_label(label: str):
    channel_min = 100
    hash_s = zlib.crc32(label.encode())
    r = ((hash_s & 0xFF0000) >> 16) % (255 - channel_min) + channel_min
    g = ((hash_s & 0x00FF00) >> 8) % (255 - channel_min) + channel_min
    b = (hash_s & 0x0000F) % (255 - channel_min) + channel_min
    return '#%02x%02x%02x' % (r, g, b)
