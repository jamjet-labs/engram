package dev.jamjet.engram.contract;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.jamjet.engram.EngramClient;
import dev.jamjet.engram.EngramConfig;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Drives every EngramClient call the Spring Boot starter makes against a live
 * Python Engram server, and asserts the response keys the Rust server returned.
 */
class EngramClientContractTest {

    private static EngramClient client() {
        return new EngramClient(EngramConfig.builder()
                .baseUrl(System.getProperty("engram.url", "http://127.0.0.1:18765"))
                .apiToken(System.getProperty("engram.token"))
                .timeoutSeconds(30)
                .build());
    }

    private static int num(Object o) {
        return ((Number) o).intValue();
    }

    @Test
    void health() {
        try (var c = client()) {
            Map<String, Object> health = c.health();
            assertEquals("ok", health.get("status"));
            assertTrue("engram".equals(health.get("service")), "health service: " + health.get("service"));
        }
    }

    @Test
    void factsRecallSearchContextAndForget() {
        String user = "facts-" + System.nanoTime();
        try (var c = client()) {
            Map<String, Object> added =
                    c.add(List.of(Map.of("role", "user", "content", "I like pizza")), user, null, null);
            assertEquals(Boolean.TRUE, added.get("success"));
            assertEquals(1, num(added.get("fact_count")));
            String factId = (String) ((List<?>) added.get("fact_ids")).get(0);

            List<Map<String, Object>> recalled = c.recall("pizza", user, null, 5);
            assertTrue(recalled.stream().anyMatch(r -> factId.equals(r.get("fact_id"))));
            for (String key : List.of("fact_id", "text", "tier", "category", "confidence")) {
                assertTrue(recalled.get(0).containsKey(key), "recall hit has " + key);
            }

            List<Map<String, Object>> searched = c.search("pizza", user, null, 5);
            assertTrue(searched.stream().anyMatch(r -> factId.equals(r.get("fact_id"))));

            Map<String, Object> ctx = c.context("pizza", user, null, 500, null);
            assertTrue(((String) ctx.get("text")).contains("pizza"));
            assertTrue(num(ctx.get("facts_included")) >= 1);
            assertTrue(ctx.containsKey("token_count"));
            assertTrue(ctx.containsKey("facts_omitted"));

            Map<String, Object> forgot = c.forget(factId, "contract test");
            assertEquals(Boolean.TRUE, forgot.get("success"));
            assertEquals(factId, forgot.get("deleted_fact_id"));
            assertFalse(c.recall("pizza", user, null, 5).stream()
                    .anyMatch(r -> factId.equals(r.get("fact_id"))), "forgotten fact is gone");

            Map<String, Object> stats = c.stats();
            for (String key : List.of("total_facts", "valid_facts", "invalidated_facts")) {
                assertTrue(stats.containsKey(key), "stats has " + key);
            }

            RuntimeException e = assertThrows(RuntimeException.class, () -> c.consolidate(user, null));
            assertTrue(e.getMessage().contains("HTTP 501"), e.getMessage());
        }
    }

    @Test
    void conversationsAndDeleteUser() {
        String user = "chat-" + System.nanoTime();
        try (var c = client()) {
            Map<String, Object> saved = c.saveMessages("conv-1",
                    List.of(Map.of("role", "user", "content", "My dog is called Rex"),
                            Map.of("role", "assistant", "content", "Nice name")),
                    user, null);
            assertEquals(Boolean.TRUE, saved.get("success"));
            assertEquals(2, ((List<?>) saved.get("message_ids")).size());
            assertTrue(saved.containsKey("fact_ids"));

            List<Map<String, Object>> msgs = c.getMessages("conv-1", null, user, null);
            assertEquals(2, msgs.size());
            assertEquals("user", msgs.get(0).get("role"));
            assertEquals("My dog is called Rex", msgs.get(0).get("content"));
            for (Map<String, Object> m : msgs) {
                for (String key : List.of("id", "conversation_id", "role", "content", "seq",
                        "created_at", "metadata")) {
                    assertTrue(m.containsKey(key), "message has " + key);
                }
            }
            assertEquals(1, c.getMessages("conv-1", 1, user, null).size());

            assertEquals(List.of("conv-1"), c.listConversations(user, null));

            Map<String, Object> deleted = c.deleteMessages("conv-1", user, null);
            assertEquals(Boolean.TRUE, deleted.get("success"));
            assertEquals(2, num(deleted.get("deleted_count")));
            assertTrue(c.getMessages("conv-1", null, user, null).isEmpty());

            c.add(List.of(Map.of("role", "user", "content", "I like tea")), user, null, null);
            Map<String, Object> gone = c.deleteUser(user);
            assertEquals(Boolean.TRUE, gone.get("success"));
            assertTrue(num(gone.get("deleted_facts")) >= 1);
            assertTrue(c.recall("tea", user, null, 5).isEmpty());
            assertTrue(c.listConversations(user, null).isEmpty());
        }
    }

    @Test
    void wrongTokenIsRefused() {
        try (var c = new EngramClient(EngramConfig.builder()
                .baseUrl(System.getProperty("engram.url", "http://127.0.0.1:18765"))
                .apiToken("wrong-token-wrong-token-wrong-token-0000")
                .build())) {
            RuntimeException e = assertThrows(RuntimeException.class, c::stats);
            assertTrue(e.getMessage().contains("Authentication failed"), e.getMessage());
        }
    }
}
