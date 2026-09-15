// SPDX-FileCopyrightText: 2025 PairInteraction Developers
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "pairinteraction/database/ParquetManager.hpp"

#include "pairinteraction/database/Database.hpp"
#include "pairinteraction/database/GitHubDownloader.hpp"

#include <doctest/doctest.h>
#include <duckdb.hpp>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <miniz.h>
#include <nlohmann/json.hpp>

namespace pairinteraction {
class MockDownloader : public GitHubDownloader {
public:
    std::future<GitHubDownloader::Result>
    download(const std::string &remote_url, const std::string & /*if_modified_since*/ = "",
             bool /*use_octet_stream*/ = false) const override {
        GitHubDownloader::Result result;
        result.status_code = 200;
        result.rate_limit.remaining = 60;
        result.rate_limit.reset_time = 2147483647;

        if (remote_url == "/test/repo/path") {
            // This is a repo path request for a single release, return JSON with assets
            result.body = make_release("1.2").dump();
        } else if (remote_url == "/test/repo/releases") {
            // This is a repo path request for a list of releases. The latest release requires a
            // newer version of the software, the older ones do not. In addition, the oldest
            // release contains an asset that is not provided anymore by newer releases.
            nlohmann::json releases = nlohmann::json::array();
            releases.push_back(make_release("2.0"));
            releases.push_back(make_release("1.2"));
            releases.push_back(make_release("1.1", {"misc", "retired"}));
            result.body = releases.dump();
        } else if (remote_url == "/rate_limit") {
            // This is the rate limit request
            result.body = "";
        } else {
            // This is the file download request
            std::string content = "updated_file_content";
            std::string filename = "misc_v1.2/wigner.parquet";

            mz_zip_archive zip_archive{};
            size_t zip_size = 0;
            void *zip_data = nullptr;

            mz_zip_writer_init_heap(&zip_archive, 0, 0);
            mz_zip_writer_add_mem(&zip_archive, filename.c_str(), content.data(), content.size(),
                                  MZ_BEST_SPEED);
            mz_zip_writer_finalize_heap_archive(&zip_archive, &zip_data, &zip_size);

            result.body = std::string(static_cast<char *>(zip_data), zip_size);

            mz_free(zip_data);
            mz_zip_writer_end(&zip_archive);
        }

        return std::async(std::launch::deferred, [result]() { return result; });
    }

private:
    // Construct a release that provides the tables of the given version for the given assets
    static nlohmann::json make_release(const std::string &version,
                                       const std::vector<std::string> &keys = {"misc"}) {
        nlohmann::json assets = nlohmann::json::array();
        for (const auto &key : keys) {
            nlohmann::json asset;
            asset["name"] = fmt::format("{}_v{}.zip", key, version);
            asset["url"] = fmt::format("https://api.github.com/test/path/{}_v{}.zip", key, version);
            assets.push_back(asset);
        }

        nlohmann::json release;
        release["assets"] = assets;
        return release;
    }
};

TEST_CASE("ParquetManager functionality with mocked downloader") {
    MockDownloader downloader;
    auto test_dir = std::filesystem::temp_directory_path() / "pairinteraction_test_db";
    std::filesystem::create_directories(test_dir / "tables" / "misc_v1.0");
    std::filesystem::create_directories(test_dir / "tables" / "misc_v1.1");
    std::ofstream(test_dir / "tables" / "misc_v1.0" / "wigner.parquet").close();
    std::ofstream(test_dir / "tables" / "misc_v1.1" / "wigner.parquet").close();
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    SUBCASE("Check missing table") {
        std::vector<std::string> repo_paths;
        ParquetManager manager(test_dir, downloader, repo_paths, con, false);
        manager.scan_local();
        manager.scan_remote();

        CHECK_THROWS_WITH_AS(manager.get_path("misc", "missing_table"),
                             "No table 'missing_table.parquet' found for species 'misc'. The "
                             "tables for the species are incomplete.",
                             std::runtime_error);
    }

    SUBCASE("Check version parsing") {
        std::vector<std::string> repo_paths;
        ParquetManager manager(test_dir, downloader, repo_paths, con, false);
        manager.scan_local();
        manager.scan_remote();

        std::string expected = (test_dir / "tables" / "misc_v1.1" / "wigner.parquet").string();
        CHECK(manager.get_path("misc", "wigner") == expected);
    }

    SUBCASE("Check update table") {
        std::vector<std::string> repo_paths = {"/test/repo/path"};
        ParquetManager manager(test_dir, downloader, repo_paths, con, false);
        manager.scan_local();
        manager.scan_remote();

        std::string expected = (test_dir / "tables" / "misc_v1.2" / "wigner.parquet").string();
        CHECK(manager.get_path("misc", "wigner") == expected);

        std::ifstream in(expected, std::ios::binary);
        std::stringstream buffer;
        buffer << in.rdbuf();
        CHECK(buffer.str() == "updated_file_content");
    }

    SUBCASE("Check update table if the latest release is incompatible") {
        std::vector<std::string> repo_paths = {"/test/repo/releases"};
        ParquetManager manager(test_dir, downloader, repo_paths, con, false);
        manager.scan_local();
        manager.scan_remote();

        // The latest compatible release must be used, not the latest release
        std::string expected = (test_dir / "tables" / "misc_v1.2" / "wigner.parquet").string();
        CHECK(manager.get_path("misc", "wigner") == expected);

        // Assets that are only provided by older releases must not be used
        CHECK_THROWS_WITH_AS(
            manager.get_path("retired", "wigner"),
            "No tables found for species 'retired'. Check the spelling of the species.",
            std::runtime_error);
    }

    std::filesystem::remove_all(test_dir);
}

DOCTEST_TEST_CASE("ParquetManager functionality with GitHub downloader") {
    if (!Database::get_global_instance().get_download_missing()) {
        DOCTEST_MESSAGE("Skipping test because download_missing is false.");
        return;
    }
    GitHubDownloader downloader;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);

    std::vector<std::string> repo_paths = {
        "/repos/pairinteraction/database-sqdt/releases?per_page=100",
        "/repos/pairinteraction/database-mqdt/releases?per_page=100"};
    ParquetManager manager(Database::get_global_instance().get_database_dir(), downloader,
                           repo_paths, con, Database::get_global_instance().get_use_cache());
    manager.scan_local();
    manager.scan_remote();

    std::string info = manager.get_versions_info();

    // Check that all species are present
    std::vector<std::string> should_contain = {
        "Cs",        "K",         "Li",         "Na",         "Rb",         "Sr87_mqdt",
        "Sr88_sqdt", "Sr88_mqdt", "Yb171_mqdt", "Yb173_mqdt", "Yb174_mqdt", "misc"};
    for (const auto &substr : should_contain) {
        DOCTEST_CHECK(info.find(substr) != std::string::npos);
    }
}

} // namespace pairinteraction
