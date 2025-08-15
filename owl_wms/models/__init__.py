def get_model_cls(model_id):
    if model_id == "game_rft":
        """
        GameRFT is Rectified Flow Transformer for video only
        """
        from .gamerft import GameRFT
        return GameRFT
    if model_id == "game_rft_audio":
        """
        GameRFTAudio is Rectified Flow Transformer for video + audio
        """
        from .gamerft_audio import GameRFTAudio
        return GameRFTAudio
    if model_id == "game_mft_audio":
        """
        GameMFTAudio is Mean Flow Transformer for video + audio
        """
        from .gamemft_audio import GameMFTAudio
        return GameMFTAudio
    if model_id == "tekken_rft":
        """
        TekkenRFT is a specialized model for the Tekken dataset.
        """
        from .tekken_rft import TekkenRFT
        return TekkenRFT
