use std::{error::Error, sync::Arc};

use actix_web::web::Bytes;
use futures::Stream;
use tokio::sync::mpsc::{channel, Receiver, Sender};

pub(crate) type SenderError = Arc<dyn Error + Send + Sync>;

pub(crate) fn new_streaming_conn() -> (Sender<Result<Bytes, SenderError>>, Client) {
    let (tx, rx) = channel(128);
    (tx, Client(rx))
}

pub(crate) struct Client(Receiver<Result<Bytes, SenderError>>);

impl Stream for Client {
    type Item = Result<Bytes, SenderError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.poll_recv(cx)
    }
}
