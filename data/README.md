# Data directory

- `raw/` contains the committed CSV inputs used by the notebooks and Streamlit app.
- `images/` is reserved for optional downloaded image datasets and is ignored by Git.

Keep credentials, personally identifiable information, and large image archives out of the repository.


## Image dataset layout

The separate image notebook expects one class folder per label:

```text
images/
├── skin_disease/<class_name>/*.(jpg|jpeg|png|bmp|webp)
└── xray/<class_name>/*.(jpg|jpeg|png|bmp|webp)
```

The complete image datasets are not committed.
