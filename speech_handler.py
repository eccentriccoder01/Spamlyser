"""
Speech Handler Module

A friendly and robust speech recognition handler that converts speech to text.
Handles microphone input, noise adjustment, and multiple recognition engines.
"""

import speech_recognition as sr
from typing import Tuple, Any


class SpeechHandler:
    """
    A user-friendly speech recognition handler.

    This class provides an easy-to-use interface for converting speech to text
    with automatic error handling and fallback mechanisms.
    """

    def __init__(self):
        """Initialize the speech handler with optimized settings."""
        self.recognizer: Any = sr.Recognizer()
        self._setup_recognizer()

    def _setup_recognizer(self) -> None:
        """Configure the recognizer with optimal settings for better accuracy."""
        # Adjust energy threshold for better speech detection
        if hasattr(self.recognizer, "energy_threshold"):
            self.recognizer.energy_threshold = 300

        # Enable dynamic energy threshold for adaptive listening
        if hasattr(self.recognizer, "dynamic_energy_threshold"):
            self.recognizer.dynamic_energy_threshold = True

        # Set pause threshold for natural speech breaks
        if hasattr(self.recognizer, "pause_threshold"):
            self.recognizer.pause_threshold = 0.8

    def _validate_audio(self, audio) -> bool:
        """Validate that the captured audio is suitable for processing."""
        if not audio or not hasattr(audio, 'sample_rate'):
            return False

        # Check if audio is too short (less than 0.5 seconds)
        if len(audio.frame_data) / audio.sample_rate < 0.5:
            return False

        return True

    def listen_and_transcribe(self, timeout: int = 5, max_duration: int = 10) -> Tuple[bool, str]:
        """
        Listen for speech and convert it to text.

        Args:
            timeout: Maximum time to wait for speech (seconds)
            max_duration: Maximum duration of speech to record (seconds)

        Returns:
            Tuple of (success: bool, result: str)
        """
        try:
            # Create a new microphone instance for fresh capture
            microphone = sr.Microphone()

            with microphone as source:
                print("🎤 Adjusting for ambient noise... Please wait.")
                # Adjust for ambient noise to improve recognition
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                try:
                    print("🎤 Listening... Please speak clearly.")
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=max_duration
                    )

                    # Validate the captured audio
                    if not self._validate_audio(audio):
                        return False, "❌ Audio too short or invalid. Please try again."

                except sr.WaitTimeoutError:
                    return False, "⏰ No speech detected. Please try speaking again."
                except Exception as e:
                    return False, f"❌ Error recording audio: {str(e)}"

            # Convert speech to text with fallback mechanisms
            return self._transcribe_audio(audio)

        except Exception as e:
            return False, f"💥 Critical error: {str(e)}"

    def _transcribe_audio(self, audio) -> Tuple[bool, str]:
        """Transcribe the captured audio to text using multiple engines."""
        try:
            # Primary: Try Google's speech recognition (most accurate)
            try:
                print("🔄 Processing with Google Speech Recognition...")
                result = self.recognizer.recognize_google(audio)
                if result and result.strip():
                    print(f"✅ Recognized: '{result}'")
                    return True, result
            except (sr.UnknownValueError, sr.RequestError) as e:
                print(f"⚠️ Google recognition failed: {str(e)}")

            # Fallback: Try local Sphinx recognizer if available
            try:
                print("🔄 Trying offline recognition...")
                result = self.recognizer.recognize_sphinx(audio)
                if result and result.strip():
                    print(f"✅ Offline recognition: '{result}'")
                    return True, result
            except ImportError:
                print("ℹ️ Offline recognizer not available")
            except Exception as e:
                print(f"⚠️ Offline recognition failed: {str(e)}")

            return False, "❓ Could not understand the audio. Please speak more clearly."

        except Exception as e:
            return False, f"❌ Speech recognition error: {str(e)}"

    def cleanup(self) -> None:
        """Clean up resources when done."""
        try:
            if hasattr(self, 'recognizer'):
                del self.recognizer
                print("🧹 Speech handler cleaned up successfully")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {str(e)}")


# Example usage
if __name__ == "__main__":
    print("🎤 Speech Handler Demo")
    print("=" * 50)

    handler = SpeechHandler()

    try:
        success, text = handler.listen_and_transcribe(timeout=5, max_duration=10)

        if success:
            print(f"\n🎉 Success! Transcribed text: '{text}'")
        else:
            print(f"\n❌ Failed: {text}")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    finally:
        handler.cleanup()
