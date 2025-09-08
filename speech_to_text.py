"""
🎤 Speech to Text Module

A friendly and robust speech recognition system with real-time processing capabilities.
Supports both continuous listening and one-time transcription with callback support.

Features:
- 🎯 Real-time continuous listening
- 🔄 One-time transcription
- 🛡️ Comprehensive error handling
- 🎨 User-friendly interface
- ⚡ Multi-threaded processing
"""

import speech_recognition as sr
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any


class SpeechToText:
    """
    🎤 Friendly Speech-to-Text Converter

    This class provides an easy-to-use interface for converting speech to text
    with both real-time continuous listening and one-time transcription capabilities.

    Features:
    - Real-time speech recognition with callbacks
    - Single utterance transcription
    - Robust error handling with user-friendly messages
    - Multi-threaded processing for smooth performance
    - Automatic microphone calibration
    """

    def __init__(self):
        """Initialize the speech-to-text system with optimal settings."""
        print("🚀 Initializing Speech-to-Text system...")

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.error_callback: Optional[Callable] = None
        self.text_callback: Optional[Callable] = None
        self.processing_thread: Optional[threading.Thread] = None

        # Configure optimal settings
        self._setup_recognizer()
        print("✅ Speech-to-Text system ready!")

    def _setup_recognizer(self) -> None:
        """Configure the recognizer with optimal settings for better accuracy."""
        try:
            print("🎤 Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Microphone calibrated successfully!")

                # Set optimal recognition parameters
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8

        except OSError as e:
            print(f"⚠️ Microphone access error: {str(e)}")
            print("💡 Please check your microphone permissions and try again.")
        except Exception as e:
            print(f"⚠️ Microphone setup warning: {str(e)}")
            print("🔧 Using default microphone settings.")

    def set_callbacks(self, text_callback: Callable, error_callback: Callable) -> None:
        """
        Set callback functions for handling transcription results and errors.

        Args:
            text_callback: Function to call when text is recognized
            error_callback: Function to call when an error occurs
        """
        self.text_callback = text_callback
        self.error_callback = error_callback
        print("📞 Callbacks configured successfully!")

    def start_listening(self) -> None:
        """Start continuous listening in a background thread."""
        if not self.is_listening:
            self.is_listening = True
            print("🎤 Starting continuous listening...")
            threading.Thread(target=self._listen_loop, daemon=True).start()
        else:
            print("ℹ️ Already listening!")

    def stop_listening(self) -> None:
        """Stop the continuous listening loop."""
        if self.is_listening:
            self.is_listening = False
            print("⏹️ Stopped listening.")
        else:
            print("ℹ️ Not currently listening.")

    def _listen_loop(self) -> None:
        """Main listening loop that runs in a separate thread."""
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=10
                    )
                    self.audio_queue.put(audio)

                    # Process audio in a separate thread to avoid blocking
                    threading.Thread(
                        target=self._process_audio,
                        args=(audio,),
                        daemon=True
                    ).start()

            except sr.WaitTimeoutError:
                continue  # No speech detected, continue listening
            except Exception as e:
                if self.error_callback:
                    self.error_callback(f"❌ Error capturing audio: {str(e)}")
                time.sleep(1)  # Prevent tight loop on error

    def _process_audio(self, audio) -> None:
        """Process captured audio and convert to text with fallback options."""
        try:
            print("🔄 Processing audio...")
            # Try Google Speech Recognition first (most accurate)
            text = self.recognizer.recognize_google(audio)  # type: ignore

            if text and text.strip():
                print(f"✅ Recognized: '{text}'")
                if self.text_callback:
                    self.text_callback(text)
            else:
                print("⚠️ Empty recognition result")

        except sr.UnknownValueError:
            error_msg = "❓ Could not understand the audio. Please speak more clearly."
            print(error_msg)
            if self.error_callback:
                self.error_callback(error_msg)

        except sr.RequestError as e:
            error_msg = f"❌ Could not connect to Google Speech Recognition: {str(e)}"
            print(error_msg)
            print("💡 Trying offline recognition...")

            # Fallback to offline recognition if available
            try:
                text = self.recognizer.recognize_sphinx(audio)  # type: ignore
                if text and text.strip():
                    print(f"✅ Offline recognition: '{text}'")
                    if self.text_callback:
                        self.text_callback(text)
                else:
                    error_msg = "❌ Offline recognition also failed"
                    print(error_msg)
                    if self.error_callback:
                        self.error_callback(error_msg)
            except ImportError:
                error_msg = "❌ Offline recognizer not available. Please install PocketSphinx."
                print(error_msg)
                if self.error_callback:
                    self.error_callback(error_msg)
            except Exception as offline_error:
                error_msg = f"❌ Offline recognition failed: {str(offline_error)}"
                print(error_msg)
                if self.error_callback:
                    self.error_callback(error_msg)

        except Exception as e:
            error_msg = f"💥 Unexpected error processing audio: {str(e)}"
            print(error_msg)
            if self.error_callback:
                self.error_callback(error_msg)

    def transcribe_once(self, timeout: int = 5, max_duration: int = 10) -> Optional[str]:
        """
        Capture and transcribe a single utterance with user-friendly feedback.

        Args:
            timeout: Maximum time to wait for speech (seconds)
            max_duration: Maximum duration of speech to record (seconds)

        Returns:
            Transcribed text or None if failed
        """
        try:
            print("🎤 Listening for single utterance... Speak now!")
            print("(Press Ctrl+C to cancel)")

            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=max_duration
                )

            print("🔄 Processing your speech...")
            # Try Google Speech Recognition first
            text = self.recognizer.recognize_google(audio)  # type: ignore

            if text and text.strip():
                print(f"🎉 Success! Transcribed: '{text}'")
                return text
            else:
                print("⚠️ No speech content detected")
                return None

        except sr.UnknownValueError:
            print("❓ Could not understand the audio. Please speak more clearly.")
        except sr.RequestError as e:
            print(f"❌ Could not connect to Google Speech Recognition: {str(e)}")
            print("💡 Trying offline recognition...")

            # Fallback to offline recognition
            try:
                text = self.recognizer.recognize_sphinx(audio)  # type: ignore
                if text and text.strip():
                    print(f"🎉 Offline recognition successful: '{text}'")
                    return text
                else:
                    print("❌ Offline recognition also failed")
            except ImportError:
                print("❌ Offline recognizer not available. Please install PocketSphinx for offline support.")
            except Exception as offline_error:
                print(f"❌ Offline recognition failed: {str(offline_error)}")
        except sr.WaitTimeoutError:
            print("⏰ No speech detected within timeout period. Please try again.")
        except KeyboardInterrupt:
            print("\n⏹️ Cancelled by user")
        except Exception as e:
            print(f"💥 Unexpected error: {str(e)}")

        return None

    def get_status(self) -> dict:
        """Get the current status of the speech recognition system."""
        return {
            "is_listening": self.is_listening,
            "queue_size": self.audio_queue.qsize(),
            "has_text_callback": self.text_callback is not None,
            "has_error_callback": self.error_callback is not None
        }


# Example usage and demo
def demo_text_callback(text: str) -> None:
    """Demo callback for successful text recognition."""
    print(f"📝 Text callback: {text}")


def demo_error_callback(error: str) -> None:
    """Demo callback for errors."""
    print(f"🚨 Error callback: {error}")


if __name__ == "__main__":
    print("🎤 Speech to Text Demo")
    print("=" * 50)

    # Create instance
    stt = SpeechToText()

    print("\nChoose an option:")
    print("1. Single transcription")
    print("2. Continuous listening demo")
    print("3. Status check")

    try:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == "1":
            print("\n--- Single Transcription Mode ---")
            result = stt.transcribe_once()
            if result:
                print(f"\nFinal result: {result}")
            else:
                print("\nNo result obtained.")

        elif choice == "2":
            print("\n--- Continuous Listening Mode ---")
            stt.set_callbacks(demo_text_callback, demo_error_callback)
            stt.start_listening()

            print("Listening... Say something! (Press Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                stt.stop_listening()
                print("\nStopped listening.")

        elif choice == "3":
            print("\n--- Status Check ---")
            status = stt.get_status()
            for key, value in status.items():
                print(f"{key}: {value}")

        else:
            print("❌ Invalid choice. Please run again.")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"💥 Demo error: {str(e)}")

    print("\n👋 Demo completed!")
