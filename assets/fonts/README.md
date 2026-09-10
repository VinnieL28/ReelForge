# Fonts

Drop `.ttf` / `.otf` files here and they take priority over every system font,
on every platform. The renderers look for these names first:

| Use | Preferred |
|---|---|
| Minimalist Motion titles | `Montserrat-ExtraBold.ttf`, `Montserrat-Bold.ttf`, `Inter-Bold.ttf` |
| Burned captions | `Montserrat-ExtraBold.ttf` |
| Body / subtitles | `Montserrat-Regular.ttf`, `Inter-Regular.ttf` |

Without them the app falls back to whatever the OS provides — Segoe UI Black or
Arial Bold on Windows, DejaVu / Liberation / FreeSans in the Docker image. It
works either way; Montserrat just looks closer to the reference channels.

Montserrat and Inter are both SIL Open Font License, so they can be committed
here and redistributed with the project. `.gitignore` allows `assets/**`.
