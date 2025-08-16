def get_loader(data_id, batch_size, **data_kwargs):
    if data_id == "cod":
        from . import cod_latent
        return cod_latent.get_loader(batch_size, **data_kwargs)
    if data_id == "cod_s3":
        from . import s3_cod_latent
        return s3_cod_latent.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3_audio":
        from . import s3_cod_latent_audio
        return s3_cod_latent_audio.get_loader(batch_size, **data_kwargs)
    elif data_id == "cod_s3_mixed":
        from . import s3_cod_latent_mixed
        return s3_cod_latent_mixed.get_loader(batch_size, **data_kwargs)
    elif data_id == "tekken_multi":
        from . import tekken_latent_multi
        return tekken_latent_multi.get_loader(batch_size, **data_kwargs)
    elif data_id == "t3":
        from . import tekken_latent
        return tekken_latent.get_loader(batch_size, **data_kwargs)
