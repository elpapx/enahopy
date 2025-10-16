"""
Mock INEI Server Tests
=====================

Tests that verify the mock server infrastructure for reliable testing.

Author: ENAHOPY Team  
Date: 2025-01-03
"""

import json
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlparse

import pandas as pd


class MockINEIServer:
    """Mock INEI server for testing download functionality."""

    def __init__(self):
        """Initialize mock server with sample data."""
        self.sample_modules = {
            "01": {  # Características de la vivienda y del hogar
                "filename": "Enaho01-2023-100.csv",
                "data": pd.DataFrame(
                    {
                        "conglome": ["001", "002", "003"],
                        "vivienda": ["01", "01", "01"],
                        "hogar": ["1", "1", "1"],
                        "ubigeo": ["150101", "150201", "150301"],
                        "tipo_vivienda": ["Casa", "Departamento", "Casa"],
                        "material_pared": ["Ladrillo", "Concreto", "Adobe"],
                    }
                ),
            },
            "34": {  # Sumaria - Ingresos y Gastos
                "filename": "Enaho01A-2023-34.csv",
                "data": pd.DataFrame(
                    {
                        "conglome": ["001", "002", "003"],
                        "vivienda": ["01", "01", "01"],
                        "hogar": ["1", "1", "1"],
                        "ubigeo": ["150101", "150201", "150301"],
                        "ingreso_total": [2000, 1500, 2500],
                        "gasto_total": [1800, 1200, 2000],
                    }
                ),
            },
            "02": {  # Características de los miembros del hogar
                "filename": "Enaho01A-2023-02.csv",
                "data": pd.DataFrame(
                    {
                        "conglome": ["001", "001", "002", "003"],
                        "vivienda": ["01", "01", "01", "01"],
                        "hogar": ["1", "1", "1", "1"],
                        "codperso": ["01", "02", "01", "01"],
                        "ubigeo": ["150101", "150101", "150201", "150301"],
                        "edad": [35, 32, 45, 28],
                        "sexo": ["M", "F", "M", "F"],
                    }
                ),
            },
        }

        self.available_years = ["2023", "2022", "2021"]
        self.request_log = []

    def get_download_url(self, year: str, module: str) -> str:
        """Generate mock download URL."""
        base_url = "http://mock-inei.gob.pe/enaho"
        filename = self.sample_modules.get(module, {}).get("filename", f"Enaho-{year}-{module}.csv")
        return f"{base_url}/{year}/{filename}"

    def mock_request(self, url: str, **kwargs) -> Mock:
        """Mock HTTP request to INEI servers."""
        self.request_log.append({"url": url, "timestamp": pd.Timestamp.now(), "kwargs": kwargs})

        # Parse URL to extract year and module
        parsed = urlparse(url)
        path_parts = parsed.path.split("/")

        if len(path_parts) >= 3:
            year = path_parts[-2]
            filename = path_parts[-1]

            # Extract module from filename
            module = self._extract_module_from_filename(filename)

            if year in self.available_years and module in self.sample_modules:
                return self._create_successful_response(module)
            else:
                return self._create_404_response(url)
        else:
            return self._create_404_response(url)

    def _extract_module_from_filename(self, filename: str) -> str:
        """Extract module code from filename."""
        # Simple extraction logic - in reality this would be more complex
        if "01-" in filename and "34" not in filename:
            return "01"
        elif "34" in filename:
            return "34"
        elif "02" in filename:
            return "02"
        else:
            # Try to extract from filename pattern
            for module in self.sample_modules:
                if module in filename:
                    return module
            return "01"  # Default

    def _create_successful_response(self, module: str) -> Mock:
        """Create successful HTTP response."""
        mock_response = Mock()
        mock_response.status_code = 200

        # Convert DataFrame to CSV
        csv_data = self.sample_modules[module]["data"].to_csv(index=False)
        mock_response.content = csv_data.encode("utf-8")
        mock_response.text = csv_data
        mock_response.headers = {"content-type": "text/csv", "content-length": str(len(csv_data))}

        # Mock streaming
        def iter_content(chunk_size=1024):
            content = mock_response.content
            for i in range(0, len(content), chunk_size):
                yield content[i : i + chunk_size]

        mock_response.iter_content = iter_content
        return mock_response

    def _create_404_response(self, url: str) -> Mock:
        """Create 404 HTTP response."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.content = f"File not found: {url}".encode("utf-8")
        mock_response.headers = {"content-type": "text/plain"}
        return mock_response

    def get_request_statistics(self) -> dict:
        """Get statistics about requests made to the mock server."""
        return {
            "total_requests": len(self.request_log),
            "requests_by_url": pd.Series([req["url"] for req in self.request_log])
            .value_counts()
            .to_dict(),
            "latest_request": self.request_log[-1] if self.request_log else None,
        }


class TestMockINEIServer(unittest.TestCase):
    """Test the mock INEI server functionality."""

    def setUp(self):
        """Set up test environment."""
        self.mock_server = MockINEIServer()

    def test_mock_server_initialization(self):
        """Test that mock server initializes correctly."""
        self.assertIsInstance(self.mock_server.sample_modules, dict)
        self.assertIn("01", self.mock_server.sample_modules)
        self.assertIn("34", self.mock_server.sample_modules)
        self.assertIn("02", self.mock_server.sample_modules)

        # Verify sample data structure
        module_01 = self.mock_server.sample_modules["01"]
        self.assertIn("filename", module_01)
        self.assertIn("data", module_01)
        self.assertIsInstance(module_01["data"], pd.DataFrame)

    def test_url_generation(self):
        """Test URL generation for different modules and years."""
        url_2023_01 = self.mock_server.get_download_url("2023", "01")
        self.assertIn("2023", url_2023_01)
        self.assertIn("mock-inei.gob.pe", url_2023_01)

        url_2022_34 = self.mock_server.get_download_url("2022", "34")
        self.assertIn("2022", url_2022_34)
        self.assertIn("34", url_2022_34)

    def test_successful_mock_requests(self):
        """Test successful mock HTTP requests."""
        url = self.mock_server.get_download_url("2023", "01")
        response = self.mock_server.mock_request(url)

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])

        # Verify CSV content
        csv_content = response.content.decode("utf-8")
        self.assertIn("conglome", csv_content)
        self.assertIn("ubigeo", csv_content)
        self.assertIn("tipo_vivienda", csv_content)

    def test_404_mock_requests(self):
        """Test 404 responses for invalid requests."""
        invalid_url = "http://mock-inei.gob.pe/invalid/2025/nonexistent.csv"
        response = self.mock_server.mock_request(invalid_url)

        self.assertEqual(response.status_code, 404)
        self.assertIn("File not found", response.content.decode("utf-8"))

    def test_request_logging(self):
        """Test that requests are properly logged."""
        url1 = self.mock_server.get_download_url("2023", "01")
        url2 = self.mock_server.get_download_url("2023", "34")

        self.mock_server.mock_request(url1)
        self.mock_server.mock_request(url2)

        stats = self.mock_server.get_request_statistics()

        self.assertEqual(stats["total_requests"], 2)
        self.assertIn(url1, stats["requests_by_url"])
        self.assertIn(url2, stats["requests_by_url"])
        self.assertIsNotNone(stats["latest_request"])

    def test_module_extraction_from_filename(self):
        """Test module extraction from different filename patterns."""
        test_cases = [
            ("Enaho01-2023-100.csv", "01"),
            ("Enaho01A-2023-34.csv", "34"),
            ("Enaho01A-2023-02.csv", "02"),
            ("unknown-pattern.csv", "01"),  # Default
        ]

        for filename, expected_module in test_cases:
            extracted = self.mock_server._extract_module_from_filename(filename)
            self.assertEqual(extracted, expected_module, f"Failed for filename: {filename}")


class TestIntegrationWithMockServer(unittest.TestCase):
    """Test integration of components with mock server."""

    def setUp(self):
        """Set up integration test environment."""
        self.mock_server = MockINEIServer()

    @patch("requests.get")
    def test_download_simulation_with_mock_server(self, mock_get):
        """Test download simulation using mock server."""
        # Configure mock to use our mock server
        mock_get.side_effect = self.mock_server.mock_request

        # Simulate multiple download requests
        test_requests = [
            ("2023", "01"),
            ("2023", "34"),
            ("2023", "02"),
            ("2022", "01"),  # This should also work
        ]

        results = []
        for year, module in test_requests:
            url = self.mock_server.get_download_url(year, module)
            response = mock_get(url)
            results.append(
                {
                    "year": year,
                    "module": module,
                    "status": response.status_code,
                    "has_data": len(response.content) > 0,
                }
            )

        # Verify all requests succeeded
        for result in results:
            self.assertEqual(result["status"], 200)
            self.assertTrue(result["has_data"])

        # Verify mock was called correct number of times
        self.assertEqual(mock_get.call_count, 4)

        # Check server statistics
        stats = self.mock_server.get_request_statistics()
        self.assertEqual(stats["total_requests"], 4)

    def test_data_consistency_across_modules(self):
        """Test that data is consistent across related modules."""
        # Get data from different modules
        module_01_data = self.mock_server.sample_modules["01"]["data"]
        module_34_data = self.mock_server.sample_modules["34"]["data"]

        # They should have the same households (conglome + vivienda + hogar)
        key_cols = ["conglome", "vivienda", "hogar"]

        keys_01 = set(module_01_data[key_cols].apply(tuple, axis=1))
        keys_34 = set(module_34_data[key_cols].apply(tuple, axis=1))

        # Should have some overlap (households appear in both modules)
        overlap = keys_01.intersection(keys_34)
        self.assertGreater(len(overlap), 0, "Modules should share some household keys")

        # UBIGEO should be consistent for same households
        for key_tuple in overlap:
            mask_01 = module_01_data[key_cols].apply(tuple, axis=1) == key_tuple
            mask_34 = module_34_data[key_cols].apply(tuple, axis=1) == key_tuple

            ubigeo_01 = module_01_data.loc[mask_01, "ubigeo"].iloc[0]
            ubigeo_34 = module_34_data.loc[mask_34, "ubigeo"].iloc[0]

            self.assertEqual(ubigeo_01, ubigeo_34, f"UBIGEO mismatch for household {key_tuple}")

    def test_mock_server_performance(self):
        """Test mock server performance under load."""
        import time

        # Simulate many requests
        n_requests = 100
        start_time = time.time()

        for i in range(n_requests):
            module = ["01", "34", "02"][i % 3]
            url = self.mock_server.get_download_url("2023", module)
            response = self.mock_server.mock_request(url)
            self.assertEqual(response.status_code, 200)

        end_time = time.time()

        # Should complete quickly (< 1 second for 100 requests)
        elapsed = end_time - start_time
        self.assertLess(
            elapsed, 1.0, f"Mock server too slow: {elapsed:.2f}s for {n_requests} requests"
        )

        # Verify statistics
        stats = self.mock_server.get_request_statistics()
        self.assertEqual(stats["total_requests"], n_requests)


if __name__ == "__main__":
    unittest.main()
